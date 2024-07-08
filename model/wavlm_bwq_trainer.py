#!/usr/bin/python
#encoding=utf-8

import math
import torch
import torch.nn as nn
import sys
import editdistance as ed
from transformers import WavLMForCTC, WavLMModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2ForCTC
import torch.nn.functional as F
from torch.distributions.normal import Normal
sys.path.append('./')
from utils.data_loader_bwq import SpeechDataset, data_collator

class WavLM_ft(nn.Module):
    def __init__(self, num_class=44, model_path='/root/workspace/MDD/CTC-Attention-Mispronunciation/egs/wavlm/model/wavlm-large'):
        super(WavLM_ft, self).__init__()
        self.num_class = num_class
        self.model_path = model_path

        self.wavlm = WavLMForCTC.from_pretrained(self.model_path)
        # self.wavlm = Wav2Vec2ForCTC.from_pretrained(self.model_path)
        self.wavlm.freeze_feature_extractor()

        self.fc2 = nn.Linear(32, self.num_class)
        # self.fc = nn.Linear(32, self.num_class)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.loss_fn = nn.CTCLoss(reduction='mean')
    

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
    
    def forward(self, input_values, attention_mask, labels, target_size, input_size, spk): # x[]
        x = input_values

        x = self.wavlm(x)
        # print(x.shape)
        x = x.logits

        x = self.fc2(x)
        x = x.transpose(0,1).contiguous() #[31, 16, 43]) 后面ctc会再变维度
        out = self.log_softmax(x)
        out_len, batch_size, _ = out.size()
        input_size = (input_size * out_len).long()

        return {
                'loss':self.loss_fn(out, labels, input_size, target_size) / batch_size,
                'predictions':out,
            }
        # return x

if __name__ == '__main__':
    import argparse
    import yaml
    class Config(object):
        batch_size = 4
        dropout = 0.1
    parser = argparse.ArgumentParser(description='cnn_lstm_ctc')
    parser.add_argument('--conf', default='conf/ctc_config_finetune.yaml' , help='conf file with argument of LSTM and training')

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
    wavlm =WavLM_ft()
    for i, data in enumerate(train_loader):
        # print(data['input_values'].shape, data['labels'].shape, data['attention_mask'].shape, data["input_size"],data["target_size"])
        inputs = data["input_values"]
        input_sizes = data["input_size"]
        targets = data['labels']
        target_sizes = data["target_size"]
        atten = data['attention_mask']
        out = wavlm(inputs, atten, targets, target_sizes , input_sizes)
        print(out['label_ids'].shape)

        print(out['predictions'].shape)
        break
    
    # x = torch.randn(16,10000)

