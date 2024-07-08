#!/usr/bin/python
#encoding=utf-8

import torch
import kaldiio
import numpy as np
import sys
import json
import torchaudio

from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
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
tokenizer = Wav2Vec2PhonemeCTCTokenizer("data/vocab.json", unk_token="[UNK]", pad_token="[PAD]",do_phonemize = False)
# tokenizer = Wav2Vec2CTCTokenizer("data/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                        return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
with open('spk.json', 'r') as f:
    spk2index = json.load(f)

#padding method
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        # print(features)
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        spk_features = [feature['spk_id'] for feature in features]
        utt = [feature['utt'] for feature in features]


        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        mask = labels != -100
        labels_size = torch.sum(mask, dim=1)
        batch["target_size"] = labels_size

        transposed_labels = batch["attention_mask"]
        labels_size = torch.count_nonzero(transposed_labels, dim=1)
        batch["input_size"] = labels_size / batch['input_values'].shape[1]
        
        spk_features = [torch.tensor(x) for x in spk_features]
        spk = torch.stack(spk_features)
        batch["spk"] = spk
        batch['utt'] = utt
        # print(batch)

        return batch


    
class SpeechDataset(Dataset):
    def __init__(self, scp_path, lab_path, trans_path, wav_path, opts, is_training=False):
        self.is_training = is_training
        self.scp_path = scp_path
        self.lab_path = lab_path
        self.trans_path = trans_path
        self.wav_path = wav_path


        self.item = []

        label_dict = dict()
        with open(self.lab_path, 'r') as rf:
            line = rf.readline()
            while line:
                utt, label = line.strip().split(' ', 1)
                label_dict[utt] = label.split()
                line = rf.readline() 

        #read the transcript
        trans_dict = dict()
        with open(self.trans_path, 'r') as rf:
            line = rf.readline()
            while line:
                utt, trans = line.strip().split(' ', 1)
                trans_dict[utt] = trans.split()
                line = rf.readline() 
        
        with open(self.wav_path, 'r') as f:
            for line in f.readlines():
                utt, path = line.strip().split(' ')
                if utt.split('_')[1] == 'arctic':
                    spk = spk2index[utt.split('_')[0]]
                else:
                    spk = spk2index['TIMIT']
                if utt.endswith('.WAV'):
                    utt = utt[:-4]
                self.item.append((path, label_dict[utt],trans_dict[utt], utt, spk))
    
    def __getitem__(self, idx):
        path, target_text, trans, utt, spk = self.item[idx]
        # print(spk)

        speech_array, sr = torchaudio.load(path)

        audio = torchaudio.transforms.Resample(sr, 16_000)(speech_array).squeeze().numpy()
        duration = len(audio) / 16_000
        sampling_rate = 16_000
        audio = processor(audio, sampling_rate=sampling_rate).input_values

        with processor.as_target_processor():
            target_text = processor(target_text).input_ids

        audio = torch.from_numpy(audio[0])
        target_text = [item for sublist in target_text for item in sublist]

        return {'input_values':audio, 'labels':target_text, 'trans':trans, 'utt':utt, 'duration':duration, 'sampling_rate':sampling_rate, 'spk_id':spk}


    def __len__(self):
        return len(self.item) 
    
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

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
    for i, data in enumerate(train_loader):
        # print(data['input_values'].shape, data['labels'].shape, data['attention_mask'].shape, data["input_size"],data["target_size"])
        print(data['spk'])
        break
    # with open('spk', 'r') as rf:
    #         spk = {}
    #         lines = rf.readlines()
    #         for i, line in enumerate(lines):
    #             spk[line.strip()] = i
    # with open("spk.json", "w") as vocab_file:
    #     json.dump(spk, vocab_file)