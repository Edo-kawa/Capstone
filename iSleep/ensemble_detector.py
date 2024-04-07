import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from models import init_bn, init_layer
from functions import do_mixup

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

def process_pred(pred):
    """
    Input: (batch_size, grid_num=10, (1+1+1+3)*2=12)
    Output: (confidence, class_vector, grid_offset, duration)
    """
    batch_size, grid_num = pred.shape[:-1]
    pred_conf = torch.zeros([batch_size, grid_num*2])
    pred_cls = torch.zeros([batch_size, grid_num*2, 3])
    pred_offset = torch.zeros([batch_size, grid_num*2])
    pred_dur = torch.zeros([batch_size, grid_num*2])

    for batch_id in range(batch_size):
        for grid_id in range(grid_num):
            pred_conf[batch_id, grid_id*2] = pred[batch_id, grid_id, 2]
            pred_conf[batch_id, grid_id*2+1] = pred[batch_id, grid_id, 8]

            pred_cls[batch_id, grid_id*2] = pred[batch_id, grid_id, 3:6]
            pred_cls[batch_id, grid_id*2+1] = pred[batch_id, grid_id, 9:]

            pred_offset[batch_id, grid_id*2] = pred[batch_id, grid_id, 0]
            pred_offset[batch_id, grid_id*2+1] = pred[batch_id, grid_id, 6]

            pred_dur[batch_id, grid_id*2] = pred[batch_id, grid_id, 1]
            pred_dur[batch_id, grid_id*2+1] = pred[batch_id, grid_id, 7]

    return pred_conf, pred_cls.permute(0, 2, 1), pred_offset, pred_dur
    # return torch.from_numpy(pred_conf).float(), torch.from_numpy(pred_cls).float(), \
    #     torch.from_numpy(pred_offset).float(), torch.from_numpy(pred_dur).float()

class MeanSquaredErrorWithLogitsLoss(nn.Module):
    def __init__(self):
        super(MeanSquaredErrorWithLogitsLoss, self).__init__()
    
    def forward(self, logits, targets, coord_scale=5.0, noobj_scale=1.0):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)

        pos_id = (targets==1.0).float()
        neg_id = (targets==0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        loss = coord_scale * pos_loss + noobj_scale * neg_loss

        return loss

class EnsembleLoss(nn.Module):
    def __init__(self, classes_num=3, noobj_scale=1.0, coord_scale=5.0):
        super(EnsembleLoss, self).__init__()

        self.classes_num = classes_num
        self.noobj_scale = noobj_scale
        self.coord_scale = coord_scale

    def forward(self, pred, target):
        """
        pred: (batch_size, grid_num, (1+1+1+3)*2=12)
        target: (batch_size, grid_num, (1+1+1+1)*2=8)
        [confidence, class_id, grid_offset, duration]
        """
        batch_size, grid_num = pred.shape[:-1]
        pred_conf, pred_cls, pred_offset, pred_dur = process_pred(pred)

        gt_conf = target[:, :, [0, 4]].view(batch_size, grid_num*2, 1)
        gt_cls = target[:, :, [1, 5]].view(batch_size, grid_num*2, 1).long()
        gt_offset = target[:, :, [2, 6]].view(batch_size, grid_num*2, 1)
        gt_dur = target[:, :, [3, 7]].view(batch_size, grid_num*2, 1)

        # obj_mask = (gt_conf == 1).float()
        # noobj_mask = (gt_conf == 0).float()
        MSELoss = nn.MSELoss(size_average=False)
        CELoss = nn.CrossEntropyLoss(size_average=False)
        MSEWithLogitsLoss = MeanSquaredErrorWithLogitsLoss()
        BCEWithLogitsLoss = nn.BCEWithLogitsLoss(size_average=False)

        # conf_diff = (gt_conf - pred_conf)**2
        # offset_diff = (gt_offset - pred_offset)**2
        # dur_diff = (gt_dur**0.5 - pred_dur**0.5)**2

        # obj_conf_loss = obj_mask * conf_diff
        # noobj_conf_loss = noobj_mask * conf_diff
        # conf_loss = (obj_conf_loss.sum() + self.noobj_scale * noobj_conf_loss.sum()) / batch_size
        conf_loss = MSEWithLogitsLoss(pred_conf, gt_conf, coord_scale=self.coord_scale, noobj_scale=self.noobj_scale) / batch_size

        # obj_gt_cls = obj_mask * gt_cls
        # obj_gt_cls = torch.eye(3)[obj_gt_cls.view(-1).long()]
        # obj_pred_cls = obj_mask.repeat(1, 3) * pred_cls
        # cls_loss = 0
        # for i in range(batch_size):
        #     cls_loss += CELoss(obj_pred_cls[i], obj_gt_cls[i])
        # cls_loss /= batch_size
        cls_loss = gt_conf * CELoss(pred_cls, gt_cls) / batch_size

        # obj_offset = obj_mask * offset_diff
        # offset_loss = (self.coord_scale * obj_offset.sum()) / batch_size
        offset_loss = gt_conf * BCEWithLogitsLoss(pred_offset, gt_offset) / batch_size

        # obj_dur = obj_mask * dur_diff
        # dur_loss = (self.coord_scale * obj_dur.sum()) / batch_size
        dur_loss = gt_conf * MSELoss(pred_dur, gt_dur) / batch_size

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
                               padding=1))
                bn = out_channels
            
            else:
                self.net.append(nn.Conv1d(in_channels=out_channels, out_channels=in_channels,
                               kernel_size=3, stride=1,
                               padding=1))
                bn = in_channels
            
            self.net.append(nn.BatchNorm1d(bn))
            self.net.append(nn.ReLU())

        self.init_weight()
        
    def init_weight(self):
        self.net.apply(init_sequential)
        
    def forward(self, input, pool_size=2, pool_type='avg'):
        x = self.net(input)
        if pool_type is None:
            return x
        elif pool_type == 'max':
            x = F.max_pool1d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool1d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool1d(x, kernel_size=pool_size)
            x2 = F.max_pool1d(x, kernel_size=pool_size)
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

        self.conf_thresh = thresh1
        self.nms_thresh = thresh2
        self.classes_num = classes_num

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
                        nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
                        nn.BatchNorm1d(512),
                        nn.ReLU()
                    )
        
        self.conv5 = Conv1dBlock(in_channels=512, out_channels=1024, num_layers=5)

        self.conv6 = nn.Sequential(
                        nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(1024),
                        nn.ReLU()
                    )
        
        # 25 -> 20
        self.conv7 = nn.Sequential(
                        nn.Conv1d(in_channels=256, out_channels=128, kernel_size=6, stride=1),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=1),
                        nn.BatchNorm1d(128),
                        nn.ReLU()
                    )
        
        # Detection Head
        self.conv8 = nn.Sequential(
                        nn.Conv1d(in_channels=1280, out_channels=1024, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Conv1d(in_channels=1024, out_channels=12, kernel_size=1, stride=1),
                        nn.BatchNorm1d(12),
                        nn.ReLU()
                    )
        
        self.init_weight()
        
    def init_weight(self):
        self.conv4.apply(init_sequential)
        self.conv7.apply(init_sequential)
        self.conv8.apply(init_sequential)

    # 32 x 128 x 20 -> 32 x 256 x 10
    def reorg(self, x, stride=2):
        batch_size, classes_num, grid_num = x.shape

        # 32 x 128 x 20 -> 32 x 128 x 2 x 10
        x = x.view(batch_size, classes_num, int(grid_num / stride), stride).transpose(3, 2).contiguous()

        # 32 x 128 x 2 x 10 -> 32 x 256 x 10
        x = x.view(batch_size, int(classes_num * stride), int(grid_num / stride))
        
        return x
    
    def nms(self, offset_preds, dur_preds, scores):
        """
        Input: (grid_num*2, 1)
        Output: List
        """

        offset = torch.sigmoid(offset_preds)
        duration = torch.exp(dur_preds)

        start_time = offset - (duration / 2.)
        end_time = offset + (duration / 2.)

        order = scores.argsort(descending=True)

        keep = []                                             
        while order.size > 0:
            i = order[0]
            keep.append(i)

            left_bound = torch.max(start_time[i], start_time[order[1:]])
            right_bound = torch.min(end_time[i], end_time[order[1:]])

            inter = torch.max(1e-10, right_bound - left_bound)
            iou = inter / (dur_preds[i] + dur_preds[order[1:]] - inter)

            inds = np.where(iou <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    @torch.no_grad()
    def postprocess(self, preds):
        """
        preds: (batch_size, grid_num=10, (1+1+1+classes_num)*2=12)
        offset      [0, 6]
        duration    [1, 7]
        confidence  [2, 8]
        class       [3:6, 9:]

        Output: List (batch_size, bboxes_num<=grid_num*2, 4)
        [confidence, offset, duration, class_id]
        """
        batch_size, grid_num = preds.shape[:-1]

        conf_preds = preds[:, :, [2, 8]].view(batch_size, grid_num*2, 1)
        cls_preds = preds[:, :, [3, 4, 5, 9, 10, 11]].view(batch_size, grid_num*2, 3)
        offset_preds = preds[:, :, [0, 6]].view(batch_size, grid_num*2, 1)
        dur_preds = preds[:, :, [1, 7]].view(batch_size, grid_num*2, 1)

        grids = torch.cat((torch.arange(10).view(-1, 1), torch.arange(10).view(-1, 1)), dim=-1).view(-1, 1)
        offset_preds += grids

        scores = torch.sigmoid(conf_preds) * torch.softmax(cls_preds, dim=-1)   # (batch_size, grid_num*2, classes_num=3)

        result = []
        for batch_id in range(batch_size):
            batch_scores = scores[batch_id]
            batch_labels = torch.argmax(batch_scores, dim=1)
            batch_scores = batch_scores[(np.arange(grid_num*2), batch_labels)]

            keep = torch.where(batch_scores >= self.conf_thresh)

            keep_offset = offset_preds[keep]
            keep_dur = dur_preds[keep]
            keep_scores = batch_scores[keep]
            keep_labels = batch_labels[keep]

            keep = torch.zeros(grid_num*2)
            for class_id in range(self.classes_num):
                inds = torch.where(keep_labels == class_id)[0]
                if len(inds) == 0:
                    continue
                class_offset = keep_offset[inds]
                class_dur = keep_dur[inds]
                class_scores = keep_scores[inds]

                class_keep = self.nms(class_offset, class_dur, class_scores)
                keep[inds[class_keep]] = 1
            
            keep = torch.where(keep > 0)
            keep_bboxes = torch.cat((keep_scores[keep], keep_offset[keep], keep_dur[keep], keep_labels[keep]), dim=1)

            result.append(keep_bboxes)
        
        return result

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)
        Output: (batch_size, grid_num=10, (1+1+1+classes_num)*2=12)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        if self.training:
            x = self.spec_augmenter(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(3, 1)   # (batch_size, mel_bins, time_steps, 1)
        x = x.squeeze(3)    # (batch_size, mel_bins, time_steps)

        x = self.bn0(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv1(x, pool_size=2, pool_type='max')
        x = self.conv2(x, pool_size=2, pool_type='max')

        x1 = torch.clone(x)     # (batch_size, 256, 25)
        x1 = self.conv7(x1)     # (batch_size, 128, 20)
        x1 = self.reorg(x1)     # (batch_size, 256, 10)

        x = self.conv3(x, pool_size=2, pool_type='max')
        x = self.conv4(x)
        x = self.conv5(x, pool_size=None, pool_type=None)
        x = self.conv6(x)

        x2 = torch.clone(x)     # (batch_size, 1024, 10)

        x = torch.concat((x2, x1), dim=1)   # (batch_size, 1280, 10)
        
        x = self.conv8(x)       # (batch_size, 12, 10)

        x = x.transpose(1, 2)

        if not self.training:
            pred = self.postprocess(x)
            return x, pred
        
        return x
