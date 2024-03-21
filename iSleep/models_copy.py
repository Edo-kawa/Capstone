import numpy as np
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from functions import do_mixup, interpolate, pad_framewise_output

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class iSleepEventDetector():
    def __init__(self, sample_rate=32000, window_size=3200, hop_size=3200):
        self.sr = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.alpha = 0.25
    
    def get_rlh(self, frame):
        low_band = [self.alpha * frame[0]]
        high_band = [self.alpha * frame[0]]

        for i in range(1, len(frame)):
            low_i = low_band[i-1] + self.alpha*(frame[i] - frame[i-1])
            high_i = self.alpha*(high_band[i-1] + frame[i] - frame[i-1])
            low_band.append(low_i)
            high_band.append(high_i)

        low_band = np.array(low_band)
        high_band = np.array(high_band)

        rms_low = np.sqrt(np.sum(low_band**2)/len(frame))
        rms_high = np.sqrt(np.sum(high_band**2)/len(frame))
        
        return rms_low / rms_high

    def preprocess(self, input):
        """
        Input: (batch_size, data_length)
        Output: (batch_size * frames_num, 3)"""

        frames = []
        features = []
        for clip in input:
            frames.append(librosa.util.frame(clip, self.window_size, self.hop_size, axis=0))
        
        frames = np.concatenate(frames, axis=0)

        for frame in frames:
            # Compute the root mean square (rms), 
            # ratio of low-band to high-band energies (rlh) and variance (var) of each frame
            rms = np.sqrt(np.sum(frame**2)/len(frame))
            rlh = self.get_rlh(frame)
            var = np.var(frame)

            features.append((rms, rlh, var))

            # TODO: Compute normalized features

        return features
    
    def forward(self, input):
        """
        Input: (batch_size, data_length)
        Output: (batch_size * frames_num, classes_num)"""

        # TODO: Use sklearn.DecisionTree to classify data points

        return None

    

class Mlp(nn.Module):
    def __init__(self, sample_rate=32000, window_size=3200, hop_size=3200, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=4):
        
        super(Mlp, self).__init__()

        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.spec_augmenter = SpecAugmentation(time_drop_width=32, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64, 128, bias=True)
        self.fc2 = nn.Linear(128, 128, bias=True)
        self.fc3 = nn.Linear(128, 64, bias=True)
        self.fc_audioset = nn.Linear(64, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)
        Output: (batch_size * frames_num, classes_num)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)   # (batch_size, mel_bins, time_steps, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)   # (batch_size, 1, time_steps, mel_bins)
        
        if self.training:
            x = self.spec_augmenter(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.squeeze(1)
        
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu_(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu_(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu_(self.fc3(x))
        x = F.dropout(x, p=0.5, training=self.training)

        framewise_output = torch.sigmoid(self.fc_audioset(x))   # (batch_size, time_steps, classes_num)
        
        # clipwise_output = torch.argmax(framewise_output, dim=2) \
        #     .mode().values.unsqueeze(1)                         # (batch_size, 1)

        return framewise_output

# Leveraged and modified from the following paper
# PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Cnn(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=4):
        
        super(Cnn, self).__init__()

        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 16     # Downsampled ratio

        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.spec_augmenter = SpecAugmentation(time_drop_width=32, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)
        Output: (batch_size, frames_num, classes_num)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]
        
        x = x.transpose(1, 3)   # (batch_size, mel_bins , time_steps, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)   # (batch_size, 1, time_steps, mel_bins)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))
        # (clipwise_output, _) = torch.max(segmentwise_output, dim=1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        return framewise_output