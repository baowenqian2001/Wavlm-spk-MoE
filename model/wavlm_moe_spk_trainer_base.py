#!/usr/bin/python
#encoding=utf-8

import math
import torch
import torch.nn as nn
import sys
import editdistance as ed
import numpy as np
from transformers import WavLMForCTC, WavLMModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2ForCTC
import torch.nn.functional as F
from torch.distributions.normal import Normal
from safetensors.torch import load_file

sys.path.append('./')

from model.wavlm_bwq_trainer import WavLM_ft_
from utils.data_loader_bwq import SpeechDataset, data_collator
import wandb
from utils.loss import softmax, amsoftmax


class Attentive_Statistics_Pooling(nn.Module):
    def __init__(self, dim):
        """ASP
        Paper: Attentive Statistics Pooling for Deep Speaker Embedding
        Link: https://arxiv.org/pdf/1803.10963.pdf
        Args:
            dim (pair): the size of attention weights
        """
        super(Attentive_Statistics_Pooling, self).__init__()
        self.sap_linear = nn.Linear(dim, dim)
        self.attention = nn.Parameter(torch.FloatTensor(dim, 1))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x):
        """Computes Attentive Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames). # fixed
        Returns:
            torch.Tensor: Output tensor (#batch, dim*2)
        """
        # x = x.permute(0, 2, 1)
        h = torch.tanh(self.sap_linear(x))
        h = torch.clamp(h, min=-1e9, max=1e9)
        
        w = torch.matmul(h, self.attention).squeeze(dim=2)
        w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
        mu = torch.sum(x * w, dim=1)
        rh = torch.sqrt( ( torch.sum((x**2) * w, dim=1) - mu**2 ).clamp(min=1e-5) )
        x = torch.cat((mu, rh), 1)
        return x
    
class MyModel7(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, skp_size, num_experts=8, topk=1, noisy_gating = False, num_layers = 1, bidirectional = False, dropout_rate=0.0):
        super(MyModel7, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w_gate = nn.Linear(skp_size, num_experts)
        self.w_noise = nn.Linear(skp_size, num_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.num_experts = num_experts
        self.k = topk
        self.noisy_gating = noisy_gating
        self.experts = nn.ModuleList([nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout = dropout_rate) for i in range(self.num_experts)])
        self.fc = nn.Linear(hidden_size, output_size)
    def _gates_to_load(self, gates):


        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):

        clean_logits = self.w_gate(x)
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, x, spk, loss_coef=1e-2):
        batch_size, seq_length, dim = x.size()
        gates, load = self.noisy_top_k_gating(spk, self.training)

        device = x.device
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        inputs = [x.clone().to(device) for i in range(self.num_experts)]

        data_output = torch.zeros(batch_size, seq_length, self.hidden_size, requires_grad=True).to(x.device)
        for i in range(self.num_experts):
            # print(gates[:,i].view(batch_size,1,1).expand(-1, seq_length, self.hidden_size).size())

            data_output = data_output + self.experts[i](inputs[i]) * gates[:,i].view(batch_size,1,1).expand(-1, seq_length, self.hidden_size)

       
        return self.fc(data_output), loss

class Moe(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, skip_size, num_experts=8, topk=4, noisy_gating=True, num_layers=2, bidirectional=True, dropout=0.1):
        super(Moe, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w_gate = nn.Linear(skip_size, num_experts) # 门控
        self.w_noise = nn.Linear(skip_size, num_experts) #  
        self.softmax = nn.Softmax(dim=-1)
        self.num_experts = num_experts
        self.softplus = F.softplus
        self.k = topk
        self.noisy_gating = noisy_gating
        self.experts = nn.ModuleList([nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)for i in range(self.num_experts)])
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self,clean_values, noisy_values, noise_stddev, noisy_top_values):

        batch = clean_values.size(0)
        m = noisy_top_values.size(1) # length
        top_values_flat = noisy_top_values.flatten() #[n]

        #  如果值在 Top-k 内，对应的阈值位置。
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k 
        # threshold_if_in: 如果值在 Top-k 内，对应的阈值。
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        self.mean, self.std = 0.0, 1.0

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev) # Normal.cdf 用于计算正态分布下，随机变量小于等于某个值的概率。
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out) # is_in,True, False
        return prob


    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2): # spk
        # print(train)
        # print(x.shape)
        clean_logits = self.w_gate(x)
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon)) # ?什么作用
            noise_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noise_logits
        else:
            logits = clean_logits
        
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        # print(111, top_logits, top_indices)
        
        top_k_logits = top_logits[:, :self.k]
        # print(top_k_logits)
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        # print(top_k_gates)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates) # ? 
        
        # 负载
        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noise_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def cv_squared(self, x):
        eps = 1e-10 
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype) # 系数方差
        return x.float().var() / (x.float().mean()**2 + eps) 


    def forward(self, x, spk, loss_coef=1e-2):
        batch_size, seq_length, dim = x.shape 
        # print(self.training)
        gates, load = self.noisy_top_k_gating(spk, self.training)
        # print(gates)
        # print(gates,load)
        device = x.device
        importance = gates.sum(0)
        # print(self.cv_squared(importance), self.cv_squared(load))
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        inputs = [x.clone().to(device) for i in range(self.num_experts)]

        data_output = torch.zeros(batch_size, seq_length, self.hidden_size, requires_grad=True).to(device)
        # print(gates)
        for i in range(self.num_experts):
            # print(self.experts[i](inputs[i])[0].shape)
            expert = self.fc1(self.experts[i](inputs[i])[0])
            g = gates[:,i].view(batch_size, 1,1).expand(-1, seq_length, self.hidden_size)
            new = expert * g
            data_output = data_output + new
            # data_output = data_output + self.experts[i](inputs[i]) * (gates[:,i].view(batch_size, 1,1).expand(-1, seq_length, self.hidden_size))
        x = self.fc(data_output)
        x = x.transpose(0,1).contiguous()

        return x , loss


class speaker(nn.Module):
    def __init__(self):
        super(speaker, self).__init__()
        self.ASP = Attentive_Statistics_Pooling(dim=768*12)
        self.LayerNorm = nn.LayerNorm(768*12)
        self.bn = nn.BatchNorm1d(768*12*2)

        self.fc = nn.Linear(768*12*2, 256) 

    def forward(self, wavlm_feature):
        # print(wavlm_feature.shape)
        sp_e = self.LayerNorm(wavlm_feature)
        sp_e = self.ASP(sp_e)
        # print(sp_e.shape,111)
        sp_e = self.bn(sp_e)
        # print('sp_e',sp_e.shape)
        sp_e = self.fc(sp_e)
        # speaker_embedding = self.classifier(speaker_embedding)
        # print(speaker_embedding.shape) #[12,2048]
        return sp_e

class WavLM_ft(nn.Module):
    def __init__(self, model, model_path='/root/workspace/MDD/CTC-Attention-Mispronunciation/egs/wavlm/model/wavlm-base'):
        super(WavLM_ft, self).__init__()
        self.model_path = model_path
        # print(self.model_path)

        
        self.wavlm = WavLMForCTC.from_pretrained(self.model_path)
        # self.wavlm = model
        self.wavlm.freeze_feature_extractor()
        self.fc2 = nn.Linear(32, 128)
        # self.fc = nn.Linear(32, self.num_class)

    def forward(self, input_values, attention_mask): # x[]
        x = input_values
        x = self.wavlm(x, attention_mask, output_hidden_states=True)
        logits = x.logits
        hidden = x.hidden_states
        hidden_states_list = list(hidden[1:])
        # print(f'hidden_or_shape:{hidden[-1].shape}')
        concatenated_hidden_states = torch.cat(hidden_states_list, dim=-1) # [f,b,d]
        # print(f'concate.shape:{concatenated_hidden_states.shape}')
        # h1 = torch.mean(concatenated_hidden_states, dim=0).squeeze(0)
        # print(h1.shape)
        # if len(h1.shape) == 3:
        #     h2 = torch.mean(h1, dim=1).squeeze(1)
        # else:
        #     h2 = torch.mean(h1, dim=0).unsqueeze(0)

        x = self.fc2(logits)
        # x = x.transpose(0,1).contiguous() #[31, 16, 43]) 后面ctc会再变维度

        return x, concatenated_hidden_states

class Wavlm_spk_Moe(nn.Module):
    def __init__(self, model_path, hidden_size = 128, num_class=44, input_size = 128, spk_emb=256, experts_num=8, spk_num=7): # 654
        super(Wavlm_spk_Moe, self).__init__()
        self.num_class = num_class
        self.spk_num = spk_num
        self.acoustic = WavLM_ft(model_path)
        self.spk = speaker()
        # self.fc2 = nn.Linear(32, self.num_class)
        # self.fc_h = nn.Linear(2048*12, 4096)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.moe = Moe(input_size=input_size, hidden_size=hidden_size, output_size=num_class, skip_size=spk_emb)
        self.spk_classifier = nn.Linear(spk_emb, self.spk_num)

        self.loss_fn1 = nn.CTCLoss(reduction='mean')
        self.loss_fn2 = amsoftmax(embedding_dim=spk_emb, num_classes=spk_num)
        # self.loss_fn2 = nn.CrossEntropyLoss()


    def forward(self, input_values, attention_mask, labels, target_size, input_size, spk):
        acou_hid, hidden = self.acoustic(input_values, attention_mask)
        # hidden = self.fc_h(hidden)
        spk_emb = self.spk(hidden)
        
        
        output, loss = self.moe(acou_hid, spk_emb)
        # print(output.shape, labels.shape)
        out = self.log_softmax(output)
        out_len, batch_size, _ = out.size()
        input_size = (input_size * out_len).long()
        ctc_loss = self.loss_fn1(out, labels, input_size, target_size) 
        # spk_emb = self.spk_classifier(spk_emb)
        # spk_emb = torch.argmax(spk_emb, -1)
        spk_loss,acc = self.loss_fn2(spk_emb, spk) # .float()
        # print(acc)
        loss_all = (0.75*ctc_loss + 0.15*spk_loss + 0.1*loss) / batch_size
        # print(loss, ctc_loss)

        # wandb.log({"ctc_loss": ctc_loss/batch_size, "spk_loss": spk_loss,'spk_acc':acc,'moe_loss':loss})
        # print(spk_emb,'111')
        # print(loss,'222')

        return {
                'loss':loss_all,
                'predictions': out #(out,spk_emb,spk,loss)
            }

    
    @staticmethod
    def save_package(model, optimizer=None, decoder=None, epoch=None, loss_results=None, dev_loss_results=None, dev_cer_results=None):
        package = {
                'num_class': model.num_class,
                'state_dict': model.state_dict()
                }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if decoder is not None:
            package['decoder'] = decoder
        if epoch is not None:
            package['epoch'] = epoch
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['dev_loss_results'] = dev_loss_results
            package['dev_cer_results'] = dev_cer_results
        return package
    

if __name__ == '__main__':
    import argparse
    import yaml
    class Config(object):
        batch_size = 4
        dropout = 0.1
    parser = argparse.ArgumentParser(description='cnn_lstm_ctc')
    parser.add_argument('--conf', default='conf/ctc_config_finetune_hasTIMIT.yaml' , help='conf file with argument of LSTM and training')

    args = parser.parse_args()

    config_path = args.conf
    conf = yaml.safe_load(open(config_path, 'r'))
    opts = Config()
    for k, v in conf.items():
        setattr(opts, k, v)
    train_dataset = SpeechDataset(opts.train_scp_path, opts.train_lab_path,opts.train_trans_path,opts.train_wav_path, opts,is_training=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                    batch_size=opts.batch_size, 
                                    shuffle=opts.shuffle_train, 
                                    num_workers=opts.num_workers,
                                    collate_fn=data_collator)
    model_path_ = '/root/workspace/MDD/CTC-Attention-Mispronunciation/egs/wavlm/checkpoint/finetune/wavlm-base/checkpoint-16875/model.safetensors'

    loaded_state_dict = load_file(model_path_)
    # print(loaded_state_dict.keys())
    model_ = WavLM_ft_(model_path='/root/workspace/MDD/CTC-Attention-Mispronunciation/egs/wavlm/model/wavlm-base')#.to(device)
    model_.load_state_dict(loaded_state_dict)


    wavlm = Wavlm_spk_Moe(model_path=model_.wavlm)#.to(device)
    # wavlm = Wavlm_spk_Moe(model_path='/root/workspace/MDD/CTC-Attention-Mispronunciation/egs/wavlm/model/wavlm-base')
    for i, data in enumerate(train_loader):
        # print(data['input_values'].shape, data['labels'].shape, data['attention_mask'].shape, data["input_size"],data["target_size"])
        inputs = data["input_values"]
        input_sizes = data["input_size"]
        targets = data['labels']
        target_sizes = data["target_size"]
        atten = data['attention_mask']
        spk_id = data["spk"]
        out = wavlm(inputs, atten, targets, target_sizes , input_sizes, spk_id)
        # print(out['loss'], out['ctc'], out['moe'], out['spk'])

        print(out['predictions'][-1])
        break
    
    # x = torch.randn(16,10000)

