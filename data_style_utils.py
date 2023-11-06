import random
from typing import Any
import numpy as np
import torch
import torch.utils.data
import os
import pathlib
from tqdm import tqdm

import commons 
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cmudict
from text.symbols import symbols


def get_speaker_id(file_name):
    file_name = file_name.split('/')[-1]
    speaker_id = file_name.split('_')[0]
    
    return speaker_id


class MultiSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, hparams) -> None:
        super().__init__()
        # self.normalization = torchvision.transforms.Normalize([0.5], [0.5])
        self.speaker_dict = {}
        self.item_list = []
        
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_noise = hparams.add_noise
        self.add_blank = getattr(hparams, "add_blank", False) # improved version
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        if getattr(hparams, "cmudict_path", None) is not None:
          self.cmudict = cmudict.CMUDict(hparams.cmudict_path)
        self.stft = commons.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        
        self.prepare_data_dir(data_paths)
        
    def prepare_data_dir(self, data_path):
        
        # ignore_file = set() # Avoid identical data points
        # data_paths = data_paths.split(',')
        # for data_path in data_paths:
        print("Preparing dataset:")
        print(data_path)
        directory = pathlib.Path(data_path)
        
        mel_files = []
        for path in directory.rglob("*.mel.npy"):
            mel_files.append(str(path))
        txt_file = []
        for path in directory.rglob("*.txt"):
            txt_file.append(str(path))
            
        for mel_file in tqdm(mel_files):
            # if mel_file in ignore_file:
            #     continue
            # codec_code = np.load(codec_file)
            
            with open(mel_file.replace(".mel.npy", ".original.txt"), "r") as f:
                text = f.read()
            
            if mel_file.replace(".mel.npy", ".normalized.txt") not in txt_file:
                text_norm = text
            else:
                with open(mel_file.replace(".mel.npy", ".normalized.txt"), "r") as f:
                    text_norm = f.read()
                    
            text_norm = self.get_text(text_norm)
                
            speaker_id = get_speaker_id(mel_file)
            if speaker_id not in self.speaker_dict.keys():
                self.speaker_dict[speaker_id] = [mel_file]
            else:
                self.speaker_dict[speaker_id].append(mel_file)
                
            self.item_list.append(
                {
                    "mel_file": mel_file,
                    "text_norm": text_norm,
                    "speaker_id": speaker_id
                }
            )
            # ignore_file.update(mel_file)
            
    def get_text(self, text):
        text_norm = text_to_sequence(text, self.text_cleaners, getattr(self, "cmudict", None))
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, len(symbols)) # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm
            
    def __len__(self):
        return len(self.item_list)
    
    def __getitem__(self, idx):
        item = self.item_list[idx]
        random_sample = self.get_random_sample(item["speaker_id"], item["mel_file"])
        
        item["sample"] = random_sample
        
        return item
    
    def get_random_sample(self, speaker_id, ignore_mel) -> np.ndarray:
        sample_list = self.speaker_dict[speaker_id]
        sample_list = [mel for mel in sample_list if mel != ignore_mel]
        
        if len(sample_list) == 0:
            sample = ignore_mel
        else:
            sample = random.sample(sample_list, k=1)[0]
        
        return sample
    
    
class MultiSpeakerCollate():
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step
        
    def get_mel(self, filename):
        melspec = torch.from_numpy(np.load(filename))
        assert melspec.size(0) == 80, (
            'Mel dimension mismatch: given {}, expected {}'.format(
                melspec.size(0), 80))
        return melspec
        
    def __call__(self, batch) -> Any:
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x["text_norm"]) for x in batch]),
            dim=0, descending=True
        )
        max_input_len = input_lengths[0]
        
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]]["text_norm"]
            text_padded[i, :text.size(0)] = text
        
        # Output mel    
        mel_batch = [self.get_mel(x["mel_file"]) for x in batch]
        num_mels = mel_batch[0].size(0)
        max_target_len = max([x.size(1) for x in mel_batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded
        y_padded = torch.FloatTensor(len(mel_batch), num_mels, max_target_len)
        y_padded.zero_()
        y_lengths = torch.LongTensor(len(mel_batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = mel_batch[ids_sorted_decreasing[i]]
            y_padded[i, :, :mel.size(1)] = mel
            y_lengths[i] = mel.size(1)
        
        # Target mel
        mel_target_batch = [self.get_mel(x["sample"]) for x in batch]
        num_mels = mel_target_batch[0].size(0)
        max_target_len = max([x.size(1) for x in mel_target_batch])
        
        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        mel_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = mel_target_batch[ids_sorted_decreasing[i]]
            mel_padded[i, :, :mel.size(1)] = mel
            mel_lengths[i] = mel.size(1)
            
        return text_padded, input_lengths, y_padded, y_lengths, mel_padded, mel_lengths
        