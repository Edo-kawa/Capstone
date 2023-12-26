import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from utilities import process_data, float32_to_int16, int16_to_float32

import config

def read_train_data(data_dir, sample_rate):
    labels = config.labels

    train_samples = {'data': [], 'target': []}
    eval_samples = {'data': [], 'target': []}
    test_samples = {'data': [], 'target': []}

    for ix, lb in enumerate(labels):
        if ix == 0:
            continue

        data_path = os.path.join(data_dir, lb)
        audios_names = os.listdir(data_path)
        # audios_num = len(audios_names)
        audios_num = 50
        eval_index, test_index = int(audios_num * 0.6) + 1, int(audios_num * 0.8) + 1

        # print(audios_num)
        padder = np.array([1, 0, 0, 0])

        # for i, audio_name in enumerate(audios_names):
        for i in range(1, audios_num+1):

            audio_name = str(i).zfill(2) + '.wav'
            audio_path = os.path.join(data_path, audio_name)

            # print(i, audio_name)

            if os.path.isfile(audio_path):
                (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
                target = [0] * len(labels)
                target[ix] = 1
                target = np.array(target, dtype=int)

                (waveform, target) = process_data(waveform, target, sample_rate, padder)
            
                if i < eval_index:
                    train_samples['data'].append(float32_to_int16(waveform))
                    train_samples['target'].append(target)
                
                elif i < test_index:
                    eval_samples['data'].append(float32_to_int16(waveform))
                    eval_samples['target'].append(target)
                
                else:
                    test_samples['data'].append(float32_to_int16(waveform))
                    test_samples['target'].append(target)

    return train_samples, eval_samples, test_samples


class EventDataSet(Dataset):
    def __init__(self, sr=32000, samples=None):
        self.sr = sr
        self.samples = samples

    def __getitem__(self, index):
        waveform = self.samples['data'][index]
        target = self.samples['target'][index]

        waveform = int16_to_float32(waveform)
        waveform = self.resample(waveform)

        target = np.array(target)

        return waveform, target
    
    def __len__(self):
        return len(self.samples['data'])

    def resample(self, waveform):
        if self.sr == 32000:
            return waveform
        elif self.sr == 16000:
            return waveform[0::2]
        elif self.sample_rate == 8000:
            return waveform[0 :: 4]
        else:
            raise Exception('Incorrect sample rate!')

