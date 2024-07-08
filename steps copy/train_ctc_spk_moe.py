#!/usr/bin/python
#encoding=utf-8
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  steps/train_ctc.py 
import os
import sys
import copy
import time 
import yaml
import argparse
import shutil
import tempfile
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist 
import torch.optim.lr_scheduler as lr_scheduler

sys.path.append('./')

from model.wavlm_bwq import *
#from warpctc_pytorch import CTCLoss # use built-in nn.CTCLoss
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader

from utils.distributed_utils import clean_up, dist, init_distrubuted_mode, is_main_process, reduce_value 
from torch.nn.parallel import DistributedDataParallel as DDP    
from utils.lr_methods import warmup
# supported_rnn = {'nn.LSTM':nn.LSTM, 'nn.GRU': nn.GRU, 'nn.RNN':nn.RNN}
# supported_activate = {'relu':nn.ReLU, 'tanh':nn.Tanh, 'sigmoid':nn.Sigmoid}

parser = argparse.ArgumentParser(description='cnn_lstm_ctc')
parser.add_argument('--conf', default='conf/ctc_config_finetune.yaml' , help='conf file with argument of LSTM and training')

def seed_torch(seed=1234):
    random.seed(seed) # Python random module.	
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed) # Numpy module.
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    print('random seed has been fixed')
seed_torch() 

def compute_wer(index, input_sizes, targets, target_sizes):
    batch_errs = 0
    batch_tokens = 0
    for i in range(len(index)):
        label = targets[i][:target_sizes[i]]
        pred = []
        for j in range(len(index[i][:input_sizes[i]])):
            if index[i][j] == 0:
                continue
            if j == 0:
                pred.append(index[i][j])
            if j > 0 and index[i][j] != index[i][j-1]:
                pred.append(index[i][j])
        batch_errs += ed.eval(label, pred)
        batch_tokens += len(label)
    return batch_errs, batch_tokens

def run_epoch(epoch, model, data_loader, loss_fn, device, optimizer, print_every=20, is_training=True, use_amp=False, lr_method=None):
    if is_training:
        model.train()
        optimizer.zero_grad()
        
    else:
        model.eval()
    
    total_loss = torch.zeros(1).to(device)
    total_tokens = 0
    total_errs = 0
    # cur_loss = 0

    lr_scheduler = None 
    if epoch == 0  and lr_method == warmup : 
        warmup_factor = 1.0/1000
        warmup_iters = min(1000, len(data_loader) -1)

        lr_scheduler = warmup(optimizer, warmup_iters, warmup_factor)

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    # 创建一个梯度缩放标量，以最大程度避免使用fp16进行运算时的梯度下溢 
    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for i, data in enumerate(data_loader):
        inputs, input_sizes, targets, target_sizes,trans,trans_sizes, utt_list = data
        inputs = inputs.to(device)
        input_sizes = input_sizes.to(device)
        targets = targets.to(device)
        target_sizes = target_sizes.to(device)
        trans = trans.to(device)
        trans_sizes = trans_sizes.to(device)

        sample_num += inputs.shape[0]

        with torch.cuda.amp.autocast(enabled=enable_amp):
            out = model(inputs) # trans #[l,b,43]
            out_len, batch_size, _ = out.size()
            input_sizes = (input_sizes * out_len).long()
            loss = loss_fn(out, targets, input_sizes, target_lengths=target_sizes)
            loss /= batch_size
            # cur_loss += loss.item()
            total_loss += reduce_value(loss, average=True).detach()
            prob, index = torch.max(out, dim=-1)
            batch_errs, batch_tokens = compute_wer(index.transpose(0,1).cpu().numpy(), input_sizes.cpu().numpy(), targets.cpu().numpy(), target_sizes.cpu().numpy())
            total_errs += batch_errs
            total_tokens += batch_tokens

            # 在进程中打印平均loss
            training = "Train" if is_training else "Valid"
            lr = optimizer.param_groups[0]["lr"] if is_training else 0.002
            if is_main_process():
                info = '[epoch {} {}]: learning_rate:{:.5f}, total_wer: {:.4f}'.format(
                    epoch + 1, 
                    training,
                    lr,
                    total_errs / total_tokens
                )
                data_loader.desc = info # tqdm 成员 desc
            
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
                lr_scheduler.step()

            # if (i + 1) % print_every == 0 and is_training:
            #     print('Epoch = %d, step = %d, cur_loss = %.4f, total_loss = %.4f, total_wer = %.4f' % (epoch_id, 
            #                             i+1, cur_loss / print_every, total_loss / (i+1), total_errs / total_tokens ))
            #     cur_loss = 0
        
        if is_training:    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    # 等待所有进程计算完毕
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)

    average_loss = total_loss / (i+1)
    
    return 1-total_errs / total_tokens, average_loss

class Config(object):
    batch_size = 4
    dropout = 0.1

def main(conf):
    opts = Config()
    for k, v in conf.items():
        setattr(opts, k, v)
        print('{:50}:{}'.format(k, v))
    

    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    # 初始化各进程环境
    init_distrubuted_mode(opts) 

    device = torch.device('cuda') if opts.use_gpu else torch.device('cpu')
    opts.init_lr *= opts.world_size  # 学习率要根据并行GPU的数量进行倍增

    if opts.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        # 这是存放你要使用tensorboard显示的数据的绝对路径 
        log_path = os.path.join(' /root/tf-logs')
        print('Start Tensorboard with "tensorboard --logdir={}"'.format(log_path)) 
        if os.path.exists(log_path) is False:
            os.makedirs(log_path)
            print("tensorboard log save in {}".format(log_path))
        else:
            shutil.rmtree(log_path) #当log文件存在时删除文件夹。记得在代码最开始import shutil 
        # 实例化一个tensorboard
        tb_writer = SummaryWriter(log_path)


    #Data Loader
    vocab = Vocab(opts.vocab_file)
    train_dataset = SpeechDataset(vocab, opts.train_scp_path, opts.train_lab_path,opts.train_trans_path,opts.train_wav_path, opts,is_training=True)
    val_dataset = SpeechDataset(vocab, opts.valid_scp_path, opts.valid_lab_path, opts.valid_trans_path, opts.valid_wav_path,opts,is_training=False)
    
    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, opts.batch_size, drop_last=True)
    nw = min([os.cpu_count(), opts.batch_size if opts.batch_size > 1 else 0, 8])  # number of workers
    if opts.rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    
    # save parameters path
    save_path = os.path.join(opts.checkpoint_dir, opts.exp_name)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    train_loader = SpeechDataLoader(train_dataset, 
                                    batch_sampler=train_batch_sampler,
                                    # batch_size=opts.batch_size, 
                                    # shuffle=opts.shuffle_train, 
                                    num_workers=nw)
    dev_loader = SpeechDataLoader(val_dataset, 
                                    batch_size=1, 
                                    sampler=val_sampler,
                                    # shuffle=False, 
                                    num_workers=nw)

    # 实例化模型
    model = WavLM_ft().to(device)
    print('model')
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    if opts.rank == 0:
        torch.save(model.state_dict(), checkpoint_path)

    dist.barrier()
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True) # opts.gpu

    num_params = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
    print("Number of parameters %d" % num_params)
    for idx, m in enumerate(model.children()):
        print(idx, m)
    
    #Training
    init_lr = opts.init_lr
    num_epoches = opts.num_epoches
    end_adjust_acc = opts.end_adjust_acc
    # decay = opts.lr_decay
    # weight_decay = opts.weight_decay
    batch_size = opts.batch_size
    
    # params = { 'num_epoches':num_epoches, 'end_adjust_acc':end_adjust_acc, 'mel': opts.mel, 'seed':opts.seed,
    #             'decay':decay, 'learning_rate':init_lr, 'weight_decay':weight_decay, 'batch_size':batch_size,
    #             'feature_type':opts.feature_type, 'n_feats': opts.feature_dim }

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    loss_fn = nn.CTCLoss( reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)#, weight_decay=weight_decay
    lf = lambda x: ((1 + math.cos(x * math.pi / opts.num_epoches)) / 2) * (1 - opts.lrf) + opts.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    count = 0
    learning_rate = init_lr
    loss_best = 1000
    loss_best_true = 1000
    adjust_rate_flag = False
    stop_train = False
    adjust_time = 0
    acc_best = 0
    start_time = time.time()
    loss_results = []
    dev_loss_results = []
    dev_cer_results = []
    sample_n = 0
    
    for epoch in range(opts.num_epoches):

        # if adjust_rate_flag:
        #     learning_rate *= decay
        #     adjust_rate_flag = False
        #     for param in optimizer.param_groups:
        #         param['lr'] *= decay
        if sample_n <= 10000:
            model.wavlm.freeze_feature_encoder()
        if opts.rank == 0:
            print("Start training epoch: %d, learning_rate: %.5f" % (epoch, learning_rate))
        
        train_acc, loss = run_epoch(epoch, model, train_loader, loss_fn, device, optimizer=optimizer, print_every=opts.verbose_step, is_training=True, use_amp=False, lr_method=warmup)
        sample_n += len(data_loader)
        scheduler.step()
        loss_results.append(loss)

        # validate
        val_acc, dev_loss = run_epoch(epoch, model, dev_loader, loss_fn, device, optimizer=None, print_every=opts.verbose_step, is_training=False, use_amp=False, lr_method=None)
        if opts.rank == 0:
            print("loss on dev set is %.4f" % dev_loss)
        dev_loss_results.append(dev_loss)
        dev_cer_results.append(val_acc)

        if opts.rank == 0: 
            print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' %  (epoch + 1, loss, train_acc, val_acc))   
        if opts.tensorboard:
            tags = ["train_loss", "train_acc", "val_accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_acc, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)
        
        #adjust learning rate by dev_loss
        # if dev_loss < (loss_best - end_adjust_acc):
        #     loss_best = dev_loss
        #     loss_best_true = dev_loss
        #     adjust_rate_count = 0
        #     model_state = copy.deepcopy(model.state_dict())
        #     op_state = copy.deepcopy(optimizer.state_dict())
        # elif (dev_loss < loss_best + end_adjust_acc):
        #     adjust_rate_count += 1
        #     if dev_loss < loss_best and dev_loss < loss_best_true:
        #         loss_best_true = dev_loss
        #         model_state = copy.deepcopy(model.state_dict())
        #         op_state = copy.deepcopy(optimizer.state_dict())
        # else:
        #     adjust_rate_count = 10

        if val_acc > acc_best:
            acc_best = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, opts.model))

        # print("adjust_rate_count:"+str(adjust_rate_count))
        # print('adjust_time:'+str(adjust_time))

        # if adjust_rate_count == 10:
        #     adjust_rate_flag = True
        #     adjust_time += 1
        #     adjust_rate_count = 0
        #     if loss_best > loss_best_true:
        #         loss_best = loss_best_true
        #     model.load_state_dict(model_state)
        #     optimizer.load_state_dict(op_state)

        # if adjust_time == 8:
        #     stop_train = True
        
        time_used = (time.time() - start_time) / 60
        if opts.rank == 0:
            print("epoch %d done, dev_acc is: %.4f, train_acc is: %.4f, time_used: %.4f minutes" % (epoch + 1, val_acc, train_acc, time_used))
        

    # print("End training, best dev loss is: %.4f, acc is: %.4f" % (loss_best, acc_best))
    # model.load_state_dict(best_model_state)
    # optimizer.load_state_dict(best_op_state)
    # save_dir = os.path.join(opts.checkpoint_dir, opts.exp_name)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # best_path = os.path.join(save_dir, 'wavlm_best_model.pkl')
    # params['epoch']=count

    # torch.save(WavLM_ft.save_package(model, optimizer=optimizer, epoch=params, loss_results=loss_results, dev_loss_results=dev_loss_results, dev_cer_results=dev_cer_results), best_path)
    # print(f'already saved in: {best_path}')

    # 删除临时缓存文件
    if opts.rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    clean_up()

if __name__ == '__main__':
    args = parser.parse_args()
    try:
        config_path = args.conf
        conf = yaml.safe_load(open(config_path, 'r'))
    except:
        print("No input config or config file missing, please check.")
        sys.exit(1)
    main(conf)
