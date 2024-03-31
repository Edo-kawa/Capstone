import numpy as np
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from models import init_bn, init_layer

def init_sequential(layer):
    if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Conv2d):
        init_layer(layer)

    elif isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.BatchNorm2d):
        init_bn(layer)

def compute_IOU(c1, d1, c2, d2):
    if c2 < c1:
        c1, c2 = c2, c1
        d1, d2 = d2, d1

    intersection = c1 - c2 + (d1 + d2) / 2
    if intersection:
        conjunction = c2 - c1 + (d1 + d2) / 2
        return intersection / conjunction

    return 0

def process_label(label):
    """
    Output: grid_id, grid_offset, duration
    """
    class_id, start_time, end_time = int(label[0]), float(label[1]), float(label[2])

    center_time = (start_time + end_time) / 2
    duration = end_time - start_time

    if duration < 1e-4:
        return False

    grid_id = int(center_time)
    grid_offset = center_time - grid_id

    return grid_id, grid_offset, duration, class_id

def process_pred(pred):
    """
    Input: (batch_size, grid_num=10, (1+1+1)*2+3=9)
    Output: (confidence, class_vector, grid_offset, duration)
    """
    batch_size, grid_num = pred.shape[:-1]
    pred_conf = np.zeros([batch_size, grid_num*2])
    pred_cls = np.zeros([batch_size, grid_num*2, 3])
    pred_offset = np.zeros([batch_size, grid_num*2])
    pred_dur = np.zeros([batch_size, grid_num*2])

    for batch_id in range(batch_size):
        for grid_id in range(grid_num):
            pred_conf[batch_id, grid_id*2] = pred[batch_id, grid_id, 2]
            pred_conf[batch_id, grid_id*2+1] = pred[batch_id, grid_id, 5]

            pred_cls[batch_id, grid_id*2] = pred[batch_id, grid_id, 6:9]
            pred_cls[batch_id, grid_id*2+1] = pred[batch_id, grid_id, 6:9]

            pred_offset[batch_id, grid_id*2] = pred[batch_id, grid_id, 0]
            pred_offset[batch_id, grid_id*2+1] = pred[batch_id, grid_id, 3]

            pred_dur[batch_id, grid_id*2] = pred[batch_id, grid_id, 1]
            pred_dur[batch_id, grid_id*2+1] = pred[batch_id, grid_id, 4]

    return torch.from_numpy(pred_conf).float(), torch.from_numpy(pred_cls).float(), \
        torch.from_numpy(pred_offset).float(), torch.from_numpy(pred_dur).float()

def generate_gt(grid_num, labels):
    """
    Input: (batch_size, bbox_num, 1+1+1=3)
    Output: (confidence, class_id, grid_offset, duration)
    """
    batch_size, bbox_num = labels.shape[:-1]
    gt_conf = np.zeros([batch_size, grid_num*2])
    gt_cls = np.zeros([batch_size, grid_num*2])
    gt_offset = np.zeros([batch_size, grid_num*2])
    gt_dur = np.zeros([batch_size, grid_num*2])

    for batch_id in range(batch_size):
        for label in labels[batch_id]:
            temp = process_label(label)

            if temp:
                grid_id, grid_offset, duration, class_id = temp

                if grid_id < grid_num:
                    gt_conf[batch_id, grid_id*2] = 1.0
                    gt_conf[batch_id, grid_id*2+1] = 1.0

                    gt_cls[batch_id, grid_id*2] = class_id
                    gt_cls[batch_id, grid_id*2+1] = class_id

                    gt_offset[batch_id, grid_id*2] = grid_offset
                    gt_offset[batch_id, grid_id*2+1] = grid_offset

                    gt_dur[batch_id, grid_id*2] = duration
                    gt_dur[batch_id, grid_id*2+1] = duration
    
    return torch.from_numpy(gt_conf).float(), torch.from_numpy(gt_cls).long(), \
        torch.from_numpy(gt_offset).float(), torch.from_numpy(gt_dur).float()

class EnsembleLoss(nn.Module):
    def __init__(self, classes_num=3, noobj_scale=0.5, coord_scale=5.0):
        super(EnsembleLoss, self).__init__()

        self.classes_num = classes_num
        self.noobj_scale = noobj_scale
        self.coord_scale = coord_scale

    def forward(self, pred, labels):
        batch_size = pred.shape[0]
        pred_conf, pred_cls, pred_offset, pred_dur = process_pred(pred)
        gt_conf, gt_cls, gt_offset, gt_dur = generate_gt(labels)

        obj_mask = (gt_conf == 1).float()
        noobj_mask = (gt_conf == 0).float()
        CELoss = nn.CrossEntropyLoss(reduction='none')

        conf_diff = (gt_conf - pred_conf)**2
        offset_diff = (gt_offset - pred_offset)**2
        dur_diff = (gt_dur**0.5 - pred_dur**0.5)**2

        obj_conf_loss = obj_mask * conf_diff
        noobj_conf_loss = noobj_mask * conf_diff
        conf_loss = (obj_conf_loss.sum() + self.noobj_scale * noobj_conf_loss.sum()) / batch_size

        obj_gt_cls = obj_mask * gt_cls
        obj_pred_cls = obj_mask.unsqueeze(2).repeat(1, 1, 3) * pred_cls.permute(0, 2, 1)
        cls_loss = CELoss(obj_pred_cls, obj_gt_cls.long())

        obj_offset = obj_mask * offset_diff
        offset_loss = (self.coord_scale * obj_offset.sum()) / batch_size

        obj_dur = obj_mask * dur_diff
        dur_loss = (self.coord_scale * obj_dur.sum()) / batch_size

        loss = conf_loss + cls_loss + offset_loss + dur_loss

        return loss

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        
        super(Conv1dBlock, self).__init__()
        
        self.net = nn.Sequential()

        for i in range(num_layers):
            if i % 2 == 0:
                self.net.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False))
                bn = out_channels
            
            else:
                self.net.append(nn.Conv1d(in_channels=out_channels, out_channels=in_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False))
                bn = in_channels
            
            self.net.append(nn.BatchNorm1d(bn))
            self.net.append(nn.ReLU())

        self.init_weight()
        
    def init_weight(self):
        self.net.apply(init_sequential)
        
    def forward(self, input, pool_size=2, pool_type='avg'):
        x = self.net(input)
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

class DetectNet(nn.Module):
    def __init__(self, thresh1=0.1, thresh2=0.5, sample_rate=32000, window_size=3200, hop_size=3200, 
        mel_bins=64, fmin=50, fmax=14000, classes_num=3):

        super(DetectNet, self).__init__()

        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.bn0 = nn.BatchNorm1d(mel_bins)

        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.spec_augmenter = SpecAugmentation(time_drop_width=32, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        # Backbone
        # 100 -> 50
        self.conv1 = Conv1dBlock(in_channels=64, out_channels=128, num_layers=1)

        # 50 -> 25
        self.conv2 = Conv1dBlock(in_channels=128, out_channels=256, num_layers=3)
        
        # 25 -> 12
        self.conv3 = Conv1dBlock(in_channels=256, out_channels=512, num_layers=5)

        # 12 -> 10
        self.conv4 = nn.Sequential(
                        nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, bias=False),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        Conv1dBlock(in_channels=512, out_channels=1024, num_layers=5),
                        nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm1d(1024),
                        nn.ReLU()
                    )
        
        # 25 -> 20
        self.conv5 = nn.Sequential(
                        nn.Conv1d(in_channels=256, out_channels=128, kernel_size=6, stride=1, bias=False),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm1d(128),
                        nn.ReLU()
                    )
        
        # Detection Head
        self.conv6 = nn.Sequential(
                        nn.Conv1d(in_channels=1280, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Conv1d(in_channels=1024, out_channels=9, kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm1d(128),
                        nn.ReLU()
                    )
        
        self.init_weight()
        
    def init_weight(self):
        self.conv4.apply(init_sequential)
        self.conv5.apply(init_sequential)
        self.conv6.apply(init_sequential)

    # 32 x 128 x 20 -> 32 x 256 x 10
    def reorg(self, x, stride=2):
        batch_size, classes_num, grid_num = input.shape

        # 32 x 128 x 20 -> 32 x 128 x 2 x 10
        x = x.view(batch_size, classes_num, int(grid_num / stride), stride).transpose(3, 2).contiguous()

        # 32 x 128 x 2 x 10 -> 32 x 256 x 10
        x = x.view(batch_size, int(classes_num * stride), int(grid_num / stride))
        
        return x

    def forward(self, input):
        """
        Input: (batch_size, data_length)
        Output: (batch_size, grid_num=10, (1+1+1)*2+classes_num=9)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        if self.training:
            x = self.spec_augmenter(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(3, 1)   # (batch_size, mel_bins, time_steps, 1)
        x = x.squeeze(3)    # (batch_size, mel_bins, time_steps)

        x = self.bn0(x)
        
        x = self.conv1(x, pool_size=2, pool_type='max')
        x = self.conv2(x, pool_size=2, pool_type='max')

        x1 = torch.clone(x)     # (batch_size, 256, 25)
        x1 = self.conv5(x1)     # (batch_size, 128, 20)
        x1 = self.reorg(x1)     # (batch_size, 256, 10)

        x = self.conv3(x, pool_size=2, pool_type='max')
        x = self.conv4(x)

        x2 = torch.clone(x)     # (batch_size, 1024, 10)
        x = torch.concat((x2, x1), dim=1)   # (batch_size, 1280, 10)
        
        x = self.conv6(x)       # (batch_size, 9, 10)
        
        return x.transpose(2, 1)