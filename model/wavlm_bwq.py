#!/usr/bin/python
#encoding=utf-8

import math
import torch
import torch.nn as nn
import editdistance as ed
from transformers import WavLMForCTC, WavLMModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2ForCTC
import torch.nn.functional as F
from torch.distributions.normal import Normal

class Moe(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, skip_size, num_experts=8, topk=1, noisy_gating=False, num_layers=1, bidirectional=False, dropout=None):
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

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev) # Normal.cdf 用于计算正态分布下，随机变量小于等于某个值的概率。
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out) # is_in,True, False
        return prob


    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2): # [b,l,d]
        clean_logits = self.self.w_gate(x)
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon)) # ?什么作用
            noise_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noise_logits
        else:
            logits = clean_logits
        
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, require_grad=True)
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
        return x.float().var() / (x.float()/x.mean()**2 + eps)


    def forward(self, x, spk, loss_coef=1e-2):
        batch_size, seq_length, dim = x.shape 
        gates, load = self.noisy_top_k_gating(spk, self.training)
        device = x.device
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        inputs = [x.clone().to(device) for i in range(self.num_experts)]

        data_output = torch.zeros(batch_size, seq_length, self.hidden_size, requires_grad=True).to(device)
        for i in range(self.num_experts):
            data_output = data_output + self.experts[i](inputs[i]) * gates[:,i].view(batch_size, 1,1).expand(-1, seq_length, self.hidden_size)

        return self.fc(data_output), loss


class speaker(nn.Module):
    def __init__(self,feature):
        super(speaker, self).__init__()
        self.speakerlayer = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1)
        )

    def forward(self, wavlm_feature):
        speaker_embedding = self.speakerlayer(wavlm_feature)
        return speaker_embedding

class WavLM_ft(nn.Module):
    def __init__(self, num_classes=43, model_path='model_path'):
        super(WavLM_ft, self).__init__()
        self.num_class = num_classes
        self.model_path = model_path

        self.wavlm = WavLMForCTC.from_pretrained(self.model_path,ctc_loss_reduction="mean")
        self.wavlm.freeze_feature_extractor()
        self.layer = nn.Sequential(
            nn.Linear(32, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.4)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Linear(32, self.num_class)
        # self.fc = nn.Linear(32, self.num_class)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    

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
    
    def forward(self, x): # x[]
        # print(f'x:{x.shape}')
        # print(f'inputs:{x.logits.shape}')
        # x = self.processor(x)
        # print(x)
        x = self.wavlm(x)
        # print(x.shape)
        x = x.logits
        # x = x.last_hidden_state
        # print(f'x{x.shape}')#[64, 789, 32])
        # x = self.layer(x)
        # x = self.layer2(x)
        x = self.fc2(x)
        x = x.transpose(0,1).contiguous() #[31, 16, 43]) 后面ctc会再变维度
        x = self.log_softmax(x)
        return x

class Wavlm_spk_Moe(nn.Module):
    def __init__():
        pass

if __name__ == '__main__':
    import torchaudio
    wavlm =WavLM()
    x = torch.randn(16,10000)
    # path = '/root/autodl-tmp/TIMIT/TEST/DR3/MMJR0/SI2166.WAV.wav'
    # processor = Wav2Vec2FeatureExtractor.from_pretrained('/root/workspace/MDD/CTC-Attention-Mispronunciation/egs/wavlm/model/wavlm-base')
    # sound, _ = torchaudio.load(path)
    # print(sound)
    # print(processor(sound))
    # print(wavlm(x).shape) # ([61, 64, 43])