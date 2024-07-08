import re
import datasets
import torch
import sys
import os
sys.path.append('./')

from safetensors.torch import load_file
import torchaudio
import itertools
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2FeatureExtractor
import soundfile as sf
import librosa
from model.wavlm_moe_spk_trainer_large import *
# from model.wavlm_bwq_trainer import WavLM_ft
from utils.data_loader_bwq import  SpeechDataset, data_collator
import random
from transformers import Wav2Vec2PhonemeCTCTokenizer

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
# output_models_dir = '/data/private/szjs8402/workspace/wav2vex_mispron/output/experiment3'

# test_dataset = datasets.load_from_disk("data/test.dataset")
tokenizer = Wav2Vec2PhonemeCTCTokenizer("data/vocab.json", unk_token="[UNK]]", pad_token="[PAD]",do_phonemize = False)
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
device = torch.device('cuda')

test_dataset = SpeechDataset(opts.test_scp_path, opts.test_lab_path,opts.test_trans_path,opts.test_wav_path, opts,is_training=False)
test_loader = torch.utils.data.DataLoader(test_dataset, 
                                batch_size=opts.batch_size, 
                                shuffle=False, 
                                num_workers=opts.num_workers,
                                collate_fn=data_collator)
path_ = '/root/workspace/MDD/CTC-Attention-Mispronunciation/egs/wavlm/checkpoint/Wavlm-moe-spk-noTIMITWavlm_large_pretrain_hasTIMIT/checkpoint-40480'
model_path = path_+'/model.safetensors'

loaded_state_dict = load_file(model_path)
model = Wavlm_spk_Moe(model_path='/root/workspace/MDD/CTC-Attention-Mispronunciation/egs/wavlm/model/wavlm-large').to(device)
# model = WavLM_ft().to(device)
model.load_state_dict(loaded_state_dict)
# print(model)

# model = Wav2Vec2ForCTC.from_pretrained(f"{output_models_dir}/checkpoint-1500")
# model.to("cuda")
def show_random_elements(dataset, num_examples=1):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    return picks
def evaluate():
    """

    :rtype: object
    """
    path = './results/' + 'Wavlm_large_hasTIMIT-'+path_.split('-')[-1]#opts.model
    if os.path.exists(path): ##目录存在，返回为真
        print( 'dir exists' )
    else:
        os.makedirs(path)
    os.system('cp -r ./result/* '+path)
    

    w1 = open(path + '/human_seq','w+') 
    w2 = open(path + '/hyp','w+')
    # inputs = processor(batch["audio"], sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        for data in test_loader:

            inputs = data['input_values'].to(device)
            # print(inputs.shape)
            input_sizes = data['input_size'].to(device)
            targets = data['labels'].to(device)
            target_sizes  = data['target_size'].to(device)
            attention_mask = data['attention_mask'].to(device)
            spk = data['spk'].to(device)
            utt = data['utt']
            logits = model(input_values=inputs, attention_mask=attention_mask, labels=targets, target_size=target_sizes, input_size=input_sizes, spk=spk)
            logits = logits['predictions']
            logits = logits.transpose(0,1).contiguous()
            
            pred_ids = torch.argmax(logits, dim=-1)
            # print(pred_ids)
            pred = tokenizer.batch_decode(pred_ids, group_tokens=False)
            target_str = tokenizer.batch_decode(targets,  group_tokens=False)
            decoded_nosil = []
            labesl_nosil = []
            # print(pred, len(target_str))
            for x in range(len(target_str)): # batch
                hyp = pred[x].split()
                # print(hyp)
                ref = target_str[x].split()
                pred_ = [ i   for i in hyp if(i != "sil" and i != "[UNK]]" and i !="blank")  ]
                pred_ = [key for key, group in itertools.groupby(pred_)]
                target = [ i   for i in ref if(i != "sil" and i != "[UNK]]")  ]
                labesl_nosil.append(' '.join(target))
                decoded_nosil.append(' '.join(pred_))  
            for x in range(len(target_str)):
                # print(utt[x])
                w1.write(utt[x] + " " + labesl_nosil[x] + "\n")  
                w2.write(utt[x] + " " + decoded_nosil[x] + "\n") 
                # print('1')
    os.chdir(path)
    os.system('chmod +777 mdd_result.sh')
    os.system('./mdd_result.sh > exp')
    w1.close()
    w2.close() 
            


    
#     batch["transcription"] = tokenizer.batch_decode(pred_ids)[0]
#     batch["names"] = batch['id']
#     return batch
# def main():

    # result = test_dataset.map(evaluate, remove_columns=test_dataset.column_names)
    # test_dataset.save_to_disk(f"data/result.dataset")

    # with open('pred.txt', 'w+') as f:
    #     for i in range(len(result)):

    #         f.write(str(result['names'][i]) + ' ' + str(result['transcription'][i]) + '\n')
    # pass
if __name__=="__main__":
    evaluate()




