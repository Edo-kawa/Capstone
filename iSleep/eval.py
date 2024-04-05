import numpy as np
import torch
from scipy import ndimage
from sklearn import metrics
from functions import (move_data_to_device, generate_framewise_target)

def get_consecutive_frames(framewise_output, class_id=None):
    if class_id is None:
        data = np.flatnonzero(framewise_output != 0)
    else:
        data = np.flatnonzero(framewise_output == class_id)
        
    return np.split(data, np.where(np.diff(data) != 1)[0]+1)

def framewise_misclassification_handling(framewise_outputs):
    # Apply opening operator with diameter=3
    # Single or continuous event frames can be filtered out 
    # if the number of these continuous frames is less than the operator diameter
    opening_result = ndimage.binary_erosion(framewise_outputs, structure=np.ones((1,5)))
    opening_result = ndimage.binary_dilation(opening_result, structure=np.ones((1,5)))
    
    # Apply closing operator with diameter=3
    # to connect those event areas with narrow gaps between them
    closing_result = ndimage.binary_dilation(opening_result, structure=np.ones((1,5)))
    closing_result = ndimage.binary_erosion(closing_result, structure=np.ones((1,5)))

    # Apply dilation operator with diameter=5
    # This will result in an expansion of 1 frame on both ends of the event frame sequences
    dilation_result = ndimage.binary_dilation(closing_result, structure=np.ones((1,5)))
    mask = dilation_result * np.ones(dilation_result.shape)
    
    for i, framewise_output in enumerate(framewise_outputs):
        cur_masks = get_consecutive_frames(mask[i])
        
        if len(cur_masks[0]) > 0:
            
            for cur_mask in cur_masks:
                class_id = np.argmax(np.bincount(framewise_output[cur_mask]))
                framewise_output[cur_mask] = class_id
            
            framewise_outputs[i] = framewise_output

    return framewise_outputs

def compute_detection_accuracy(preds, targets):
    samples_num = preds.shape[0]
    preds = framewise_misclassification_handling(np.int64(preds))

    move_FDA_up = move_FDA_down = cough_EDA_up = cough_EDA_down = snoring_EDA_up = snoring_EDA_down = 0

    for sample_id in range(samples_num):
        pred, target = preds[sample_id], targets[sample_id]

        # Compute FDA for frames related to 'move'
        move_target_num = (target == 2).sum()
        move_pred_num = (pred[target == 2] == 2).sum()
        
        move_FDA_up += move_pred_num
        move_FDA_down += move_target_num

        # Compute EDA for frames related to 'snoring' and 'cough'
        cough_target_ind = get_consecutive_frames(target, 1)
        snoring_target_ind = get_consecutive_frames(target, 3)
        cough_EDA_down += len(cough_target_ind)
        snoring_EDA_down += len(snoring_target_ind)
        
        for cough_event in cough_target_ind:
            cough_EDA_up += 1 if 1 in pred[cough_event] else 0
        
        for snoring_event in snoring_target_ind:
            snoring_EDA_up += 1 if 3 in pred[snoring_event] else 0

    return move_FDA_up, move_FDA_down, cough_EDA_up, cough_EDA_down, snoring_EDA_up, snoring_EDA_down

def evaluate(model, data_loader):
    """Args:
    model: torch.nn.Module
    data_loader: torch.utils.data.DataLoader
    """

    device = next(model.parameters())
    
    result = [0] * 6

    for i, (waveform, target) in enumerate(data_loader):
        waveform, targets = move_data_to_device(waveform, device), move_data_to_device(target, device)

        with torch.no_grad():
            model.eval()
            framewise_outputs = model(waveform, None)

        
        avg_precision = metrics.average_precision_score(targets.view(-1, 4).cpu(), framewise_outputs.view(-1, 4).cpu(), average=None)
        roc_auc = metrics.roc_auc_score(targets.view(-1, 4).cpu(), framewise_outputs.view(-1, 4).cpu(), average=None)

        framewise_outputs = torch.argmax(framewise_outputs, dim=2).cpu().numpy()
        targets = torch.argmax(targets, dim=2).cpu().numpy()

        move_FDA_up, move_FDA_down, cough_EDA_up, cough_EDA_down, \
            snoring_EDA_up, snoring_EDA_down = compute_detection_accuracy(framewise_outputs, targets)
            
        result[0] += move_FDA_up; result[1] += move_FDA_down
        result[2] += cough_EDA_up; result[3] += cough_EDA_down
        result[4] += snoring_EDA_up; result[5] += snoring_EDA_down
        

    statistics = {'move_FDA_up': result[0], 'move_FDA_down': result[1],
                  'cough_EDA_up': result[2], 'cough_EDA_down': result[3],
                  'snoring_EDA_up': result[4], 'snoring_EDA_down': result[5],
                  'average_precision': avg_precision, 'roc_auc_score': roc_auc}

    return statistics

def evaluate_ensemble(model, data_loader):
    """Args:
    model: torch.nn.Module
    data_loader: torch.utils.data.DataLoader
    
    pred:
    offset      [0, 6]
    duration    [1, 7]
    confidence  [2, 8]
    class       [3:6, 9:]
    """

    device = next(model.parameters()).device

    for i, (waveform, target) in enumerate(data_loader):
        waveform, target = move_data_to_device(waveform, device), move_data_to_device(target, device)

        with torch.no_grad():
            model.eval()
            ensemble_output = model(waveform, None)

    # output: (batch_size, 10, 12)
    # target: (batch_size, 10, 8)

    conf_outputs = ensemble_output[:, :, [2, 8]].view(-1, 1).cpu()
    cls_outputs = ensemble_output[:, :, [3, 4, 5, 9, 10, 11]].view(-1, 3).cpu()
    dur_outputs = ensemble_output[:, :, [1, 7]].view(-1, 1).cpu()
    offset_outputs = ensemble_output[:, :, [0, 6]].view(-1, 1).cpu()

    cls_target = target[:, :, [1, 5]].view(-1).int().cpu()
    cls_targets = torch.eye(3)[cls_target].cpu()
    dur_targets = target[:, :, [3, 7]].view(-1).float().cpu()

    avg_precision = metrics.average_precision_score(cls_targets, cls_outputs, average=None)
    roc_auc = metrics.roc_auc_score(cls_targets, cls_outputs, average=None)

    dur_diff = torch.abs(dur_outputs - dur_targets)
    dur_score = torch.mean(dur_diff)

    statistics = {'average_precision': avg_precision, 'roc_auc': roc_auc, 'dur_score': dur_score}

    return statistics