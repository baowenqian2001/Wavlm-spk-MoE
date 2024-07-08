#!/usr/bin/python
#encoding=utf-8
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4  steps/train_ctc.py 
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
import wandb
import torch
import torch.nn as nn
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WavLMForCTC
)
from datasets import load_dataset, load_metric
import torch.optim.lr_scheduler as lr_scheduler
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
sys.path.append('./')
import sklearn
from utils.loss import amsoftmax
from utils.lr_methods import warmup
from model.wavlm_moe_spk_trainer_large import *
#from warpctc_pytorch import CTCLoss # use built-in nn.CTCLoss
from utils.data_loader_bwq import SpeechDataset, data_collator
from utils.distributed_utils import clean_up, dist, init_distrubuted_mode, is_main_process, reduce_value 
# supported_rnn = {'nn.LSTM':nn.LSTM, 'nn.GRU': nn.GRU, 'nn.RNN':nn.RNN}
# supported_activate = {'relu':nn.ReLU, 'tanh':nn.Tanh, 'sigmoid':nn.Sigmoid}

wandb.init(
    # set the wandb project where this run will be logged
    project="huggingface",
    name='"Wavlm_large_pretrain_hasTIMIT',
    
    # track hyperparameters and run metadata
    config={
    "epochs": 50,
    }
)

# wer_metric = load_metric("utils/wer.py")
parser = argparse.ArgumentParser(description='cnn_lstm_ctc')
parser.add_argument('--conf', default='conf/ctc_config_finetune_hasTIMIT.yaml' , help='conf file with argument of LSTM and training')

tokenizer = Wav2Vec2PhonemeCTCTokenizer("data/vocab.json", unk_token="[UNK]", pad_token="[PAD]",do_phonemize = False)
# tokenizer = Wav2Vec2CTCTokenizer("data/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                        return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# old_f = sys.stdout
# class F:
#     def write(self, x):
#         old_f.write(x.replace("\n", " [%s]\n" % str(traceback.extract_stack())))
# sys.stdout = F()

def seed_torch(seed=1234):
    random.seed(seed) # Python random module.	
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed) # Numpy module.
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    print('random seed has been fixed')
seed_torch() 


def compute_metrics(pred):
    pred_logits = pred.predictions
    out,spk_emb,spk,loss = pred_logits
    # print(spk)
    spk = torch.tensor(spk)
    # print(spk)
    spk_emb = torch.tensor(spk_emb)
    # print(spk_emb.argmax(-1))
    loss_fn2 = amsoftmax(256, num_classes=7)
    spk_loss,acc = loss_fn2(spk_emb, spk)

    return {
        'spk_loss':spk_loss,
        'spk_acc':acc,
        'moe_loss':loss.mean()
    }


    # print(metric)
    # pred_logits = pred.predictions
    # pred_ids = np.argmax(pred_logits[0], axis=-1)
    # # pred_ids = pred_ids.transpose(0,1)
    # # print(pred_logits[0].shape,pred.label_ids.shape)
    # pred_logits[0][pred_logits[0] == -100] = processor.tokenizer.pad_token_id
    # pred_str = processor.batch_decode(pred_ids)
    # # we do not want to group tokens when computing the metrics
    # label_str = processor.batch_decode(pred_logits[0], group_tokens=False)
    # wer = wer_metric.compute(predictions=pred_str, references=label_str)
    # wandb.log({"wer": wer})
    # return {
    #     'wer':wer
    # }

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


from safetensors.torch import load_file
class Config(object):
    batch_size = 4
    dropout = 0.1

def main(conf):
    opts = Config()
    for k, v in conf.items():
        setattr(opts, k, v)
        # print('{:50}:{}'.format(k, v))
    
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
        
    device = torch.device('cuda') if opts.use_gpu else torch.device('cpu')
    # opts.init_lr *= opts.world_size  # 学习率要根据并行GPU的数量进行倍增

    #Data Loader
    # vocab = Vocab(opts.vocab_file)
    train_dataset = SpeechDataset(opts.train_scp_path, opts.train_lab_path,opts.train_trans_path,opts.train_wav_path, opts,is_training=True)

    val_dataset = SpeechDataset(opts.valid_scp_path, opts.valid_lab_path, opts.valid_trans_path, opts.valid_wav_path,opts,is_training=False)
    
    # 预训练wavlm-base
    model_path_ = '/root/workspace/MDD/CTC-Attention-Mispronunciation/egs/wavlm/checkpoint/finetune/wavlm-large/checkpoint-18590/model.safetensors'
    loaded_state_dict = load_file(model_path_)
    model_ = WavLM_ft_(model_path='/root/workspace/MDD/CTC-Attention-Mispronunciation/egs/wavlm/model/wavlm-large').to(device)
    model_.load_state_dict(loaded_state_dict)

    # print(opts.model_path)
    # model_path = '/root/workspace/MDD/CTC-Attention-Mispronunciation/egs/wavlm/checkpoint/Wavlm-moe-spk-noTIMITWavlm_base_pretrain_hasTIMIT/checkpoint-5402/model.safetensors'

    # loaded_state_dict = load_file(model_path)
    # model = Wavlm_spk_Moe().to(device)
    model = Wavlm_spk_Moe(model_path=model_.wavlm).to(device)# model_path='/root/workspace/MDD/CTC-Attention-Mispronunciation/egs/wavlm/model/wavlm-base'
    
    # model = Wavlm_spk_Moe(model_path=opts.model_path).to(device)
    # model.load_state_dict(loaded_state_dict)
    
    # class My_Trainer(Trainer):
    #     def compute_loss(self, model, inputs, return_outputs=False):
    #         input_values, attention_mask, labels, target_size, input_size, spk = inputs.values()
    #         loss_all, ctc_loss, spk_loss, loss = model(input_values, attention_mask, labels, target_size, input_size, spk)
    #         print(loss_all)
    #         wandb.log({"ctc_loss":ctc_loss,"spk": spk_loss, "mode_loss": loss})
    #         return loss_all


    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
   # save parameters path
    save_path = os.path.join(opts.checkpoint_dir, opts.exp_name)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    training_args = TrainingArguments(
        output_dir=save_path+"Wavlm_large_pretrain_hasTIMIT",          # output directory
        overwrite_output_dir=True,
        num_train_epochs=300,              # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        evaluation_strategy='epoch', 
        save_steps=15,  
        # label_names=['ctc','spk','moe'],
        # load_best_model_at_end=True,
        remove_unused_columns= False,
        save_total_limit=10, 
        # include_inputs_for_metrics= True, # compute_metrics 时需要原始输出来计算评价指标
        # metric_for_best_model="ctc",
        save_strategy='epoch',    # "no": No evaluation is done during training.
                                        # "steps": Evaluation is done (and logged) every steps
                                        # "epoch": Evaluation is done at the end of each epoch.
    )
    class my_trainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=True):
            input_values, attention_mask, labels, target_size, input_size, spk = inputs.values()
            loss_all, ctc_loss, spk_loss, loss = model(input_values, attention_mask, labels, target_size, input_size, spk)
            return loss_all, {'ctc_loss':ctc_loss, 'spk_loss':spk_loss, 'moe_loss':loss}
        
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        # length_field_name=data_args.length_field_name,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
        )
    trainer.train()
    trainer.evaluate()
   

    # model.save_package(model)
    # processor.save_pretrained(save_path+"wavlm-base-finetune")

    # train_loader = torch.utils.data.DataLoader(train_dataset, 
    #                                 batch_size=opts.batch_size, 
    #                                 shuffle=opts.shuffle_train, 
    #                                 num_workers=opts.num_workers,
    #                                 collate_fn=data_collator)
    # dev_loader = torch.utils.data.DataLoader(val_dataset, 
    #                                 batch_size=1, 
    #                                 shuffle=False, 
    #                                 num_workers=opts.num_workers,
    #                                 collate_fn=data_collator)

    # 实例化模型

    # print('model')
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")


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
    
    # for epoch in range(opts.num_epoches):
    #     print("Start training epoch: %d, learning_rate: %.5f" % (epoch, learning_rate))
    #     # if sample_n <= 500:
    #     #     model.module.wavlm.freeze_feature_encoder()
    #     train_acc, loss = run_epoch(epoch, model, train_loader, loss_fn, device, optimizer=optimizer, print_every=opts.verbose_step, is_training=True, use_amp=True, lr_method=warmup)
    #     scheduler.step()
    #     loss_results.append(loss)
    #     sample_n += len(train_loader)

    #     # validate
    #     val_acc, dev_loss = run_epoch(epoch, model, dev_loader, loss_fn, device, optimizer=None, print_every=opts.verbose_step, is_training=False, use_amp=False, lr_method=None)
    #     print("loss on dev set is %.4f" % dev_loss)
    #     dev_loss_results.append(dev_loss)
    #     dev_cer_results.append(val_acc)


    #     print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' %  (epoch + 1, loss, train_acc, val_acc))   
    #     if opts.tensorboard:
    #         tags = ["train_loss", "train_acc", "val_accuracy", "learning_rate"]
    #         tb_writer.add_scalar(tags[0], loss, epoch)
    #         tb_writer.add_scalar(tags[1], train_acc, epoch)
    #         tb_writer.add_scalar(tags[2], val_acc, epoch)
    #         tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)
        


    #     if val_acc > acc_best:
    #         acc_best = val_acc
    #         torch.save(model.module.save_package(model.module, optimizer=optimizer, epoch=epoch, loss_results=loss_results, dev_loss_results=dev_loss_results, dev_cer_results=dev_cer_results), os.path.join(save_path, opts.model))
        
    #     time_used = (time.time() - start_time) / 60

    #     print("epoch %d done, dev_acc is: %.4f, train_acc is: %.4f, time_used: %.4f minutes" % (epoch + 1, val_acc, train_acc, time_used))
        


    # # 删除临时缓存文件
    # if os.path.exists(checkpoint_path) is True:
    #     os.remove(checkpoint_path)



if __name__ == '__main__':
    args = parser.parse_args()
    try:
        config_path = args.conf
        conf = yaml.safe_load(open(config_path, 'r'))
    except:
        print("No input config or config file missing, please check.")
        sys.exit(1)
    main(conf)
