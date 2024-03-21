import numpy as np
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from functions import do_mixup, interpolate, pad_framewise_output
from sklearn.tree import DecisionTreeClassifier

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
    def __init__(self, classification_head=DecisionTreeClassifier()):
        self.noise_features = None
        self.classification_head = classification_head
    
    def update_noise_features(self, new_features):
        new_features -= self.noise_features
        self.noise_features += new_features * 0.5

    def framing(self, input, window_size=3200, hop_size=3200):
        """
        Input: (batch_size, data_length)
        Output: (batch_size, frames_num, frame_length)
        """
        frames = [librosa.util.frame(clip, window_size, hop_size, axis=0) for clip in input]
        return np.asarray(frames)

    def get_rlh(self, frame, alpha=0.25):
        frame_length = len(frame)
        low_band, high_band = alpha * frame[0], alpha * frame[0]
        rms_low, rms_high = low_band**2, high_band**2

        for i in range(1, len(frame)):
            # low_band[i] = low_band[i-1] + self.alpha*(frame[i] - frame[i-1])
            # high_band[i] = self.alpha*(high_band[i-1] + frame[i] - frame[i-1])
            low_band += alpha*(frame[i] - frame[i-1])
            high_band = alpha*(high_band + frame[i] - frame[i-1])
            rms_low += low_band**2
            rms_high += high_band**2

        rms_low = np.sqrt(rms_low/frame_length)
        rms_high = np.sqrt(rms_high/frame_length)
        
        return rms_low / rms_high

    def compute_noise_features(self, noise_frames, frame_length):
        noise_rms = np.sqrt(np.sum(noise_frames**2, axis=1)/frame_length)
        noise_rlh = np.array([self.get_rlh(noise_frame for noise_frame in noise_frames)])
        noise_var = np.var(noise_frames, axis=1)

        noise_features = np.array([np.mean(noise_rms), np.std(noise_rms), np.mean(noise_rlh), \
                                   np.std(noise_rlh), np.mean(noise_var), np.std(noise_var)])

        if self.noise_features is None:
            self.noise_features = noise_features
        else:
            self.update_noise_features(new_features=noise_features)
        
        return noise_features

    def compute_non_noise_features(self, non_noise_frames, frame_length):
        non_noise_rms = np.sqrt(np.sum(non_noise_frames**2, axis=1)/frame_length)
        non_noise_rlh = np.array([self.get_rlh(non_noise_frame for non_noise_frame in non_noise_frames)])
        non_noise_var = np.var(non_noise_frames, axis=1)

        non_noise_rms = (non_noise_rms - self.noise_features[0]) / self.noise_features[1]
        non_noise_rlh = (non_noise_rlh - self.noise_features[2]) / self.noise_features[3]
        non_noise_var = (non_noise_var - self.noise_features[4]) / self.noise_features[5]

        return np.array([non_noise_rms, non_noise_rlh, non_noise_var])

    def fit(self, input, targets):
        """
        input: (batch_size, data_length)
        targets: (batch_size, frames_num, classes_num)
        """

        windows = self.framing(input)    # (batch_size, frames_num, frame_length)
        number_targets = np.argmax(targets, axis=1)

        samples_num, frames_num, frame_length = windows.shape
        stds = np.std(windows, axis=2)
        
        stds_mean, stds_min = np.mean(stds, axis=1).reshape(samples_num, 1), np.min(stds, axis=1).reshape(samples_num, 1)
        stds = (stds - stds_mean) / (stds_mean - stds_min)

        vars = stds**2
        noise_frames = windows[vars < 0.5]

        _ = self.compute_noise_features(noise_frames, frame_length)

        non_noise_frames = windows[vars >= 0.5]
        non_noise_targets = number_targets[vars >= 0.5]

        non_noise_features = self.compute_non_noise_features(non_noise_frames, frame_length)

        self.classification_head.fit(non_noise_features, non_noise_targets)

        return
    
    def predict(self, input, transform=False):
        """
        Input: (batch_size, data_length)
        Output: (batch_size, frames_num, classes_num)"""

        windows = self.framing(input)
        samples_num, frames_num, frame_length = windows.shape
        stds = np.std(windows, axis=2)
        preds = np.zeros((samples_num, frames_num))
        
        stds_mean, stds_min = np.mean(stds, axis=1).reshape(samples_num, 1), np.min(stds, axis=1).reshape(samples_num, 1)
        stds = (stds - stds_mean) / (stds_mean - stds_min)

        vars = stds**2
        preds[vars < 0.5] = 0

        non_noise_frames = windows[vars >= 0.5]
        non_noise_features = self.compute_non_noise_features(non_noise_frames, frame_length)
        preds[vars >= 0.5] = self.classification_head.predict(non_noise_features)

        if transform:
            noise_frames = windows[vars < 0.5]
            noise_features = self.compute_noise_features(noise_frames, frame_length)
            self.update_noise_features(noise_features)

        return preds

    

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

        self.bn0 = nn.BatchNorm2d(mel_bins)

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



class Cnn(nn.Module):
    def __init__(self, sample_rate=32000, window_size=3200, hop_size=3200, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=4):
        
        super(Cnn, self).__init__()

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

        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.bn1 = nn.BatchNorm1d(mel_bins*2)
        self.bn2 = nn.BatchNorm1d(mel_bins*4)
        self.bn3 = nn.BatchNorm1d(mel_bins*8)

        self.conv1 = nn.Conv1d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
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

        x = x.transpose(1, 3)
        x = x.squeeze(3)    # (batch_size, mel_bins, time_steps)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        framewise_output = torch.sigmoid(self.fc_audioset(x))   # (batch_size, time_steps, classes_num)
        
        # clipwise_output = torch.argmax(framewise_output, dim=2) \
        #     .mode().values.unsqueeze(1)                         # (batch_size, 1)

        return framewise_output



# (Wavegram-Logmel-)CNN14
# Leveraged and modified from the following paper
# PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPreWavBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1,
                              padding=1, bias=False)
                              
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1, dilation=2, 
                              padding=2, bias=False)
                              
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)
        
        return x

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

class Wavegram_Logmel_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Wavegram_Logmel_Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        # Log mel spectrogram
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
            a1 = do_mixup(a1, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')

        # Concatenate Wavegram and Log mel spectrogram along the channel dimension
        x = torch.cat((x, a1), dim=1)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        # (x1, _) = torch.max(x, dim=2)
        # x2 = torch.mean(x, dim=2)
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))
        
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        return framewise_output

class Cnn14(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=4):
        
        super(Cnn14, self).__init__()

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

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
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
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)        
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        # (x1, _) = torch.max(x, dim=2)
        # x2 = torch.mean(x, dim=2)
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        return framewise_output

class EnsembleClassifier(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=4):

        super(EnsembleClassifier, self).__init__()

    def forward(self, input):
        pass