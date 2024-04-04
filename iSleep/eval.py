import numpy as np
import torch
from scipy import ndimage
from sklearn import metrics
from functions import (move_data_to_device, generate_framewise_target)

def get_consecutive_frames(framewise_output, class_id):
    data = np.flatnonzero(framewise_output == class_id)
    return np.split(data, np.where(np.diff(data) != 1)[0]+1)

def framewise_misclassification_handling(framewise_outputs, isDT=False):
    temp = framewise_outputs if isDT else torch.argmax(framewise_outputs, dim=2).numpy()

    # Apply opening operator with diameter=5
    # Single or continuous event frames can be filtered out 
    # if the number of these continuous frames is less than the operator diameter
    opening_result = ndimage.grey_erosion(temp, size=(1, 5))
    opening_result = ndimage.grey_dilation(opening_result, size=(1, 5))

    # Apply closing operator with diameter=5
    # to connect those event areas with narrow gaps between them
    closing_result = ndimage.grey_dilation(opening_result, size=(1, 5))
    closing_result = ndimage.grey_erosion(closing_result, size=(1, 5))

    # Apply dilation operator with diameter=5
    # This will result in an expansion of 2 frames on both ends of the event frame sequences
    dilation_result = ndimage.grey_dilation(closing_result, size=(1, 5))

    return dilation_result if isDT else torch.from_numpy(dilation_result)

def compute_detection_accuracy(preds, targets, isDT=False):
    samples_num = preds.shape[0]
    preds = framewise_misclassification_handling(preds, isDT=isDT)

    move_FDA_up = move_FDA_down = cough_EDA_up = cough_EDA_down = snoring_EDA_up = snoring_EDA_down = 0

    for sample_id in range(samples_num):
        pred, target = preds[sample_id], targets[sample_id]

        # Compute FDA for frames related to 'move'
        move_target_ind = (target == 2)
        move_FDA_down += move_target_ind.sum()
        move_FDA_up += (pred[move_target_ind] == 2).sum()

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

    device = next(model.parameters()).device

    for i, (waveform, target) in enumerate(data_loader):
        waveform, targets = move_data_to_device(waveform, device), move_data_to_device(target, device)

        with torch.no_grad():
            model.eval()
            framewise_outputs = model(waveform, None)

    targets = torch.argmax(targets, dim=2)

    move_FDA_up, move_FDA_down, cough_EDA_up, cough_EDA_down, \
        snoring_EDA_up, snoring_EDA_down = compute_detection_accuracy(framewise_outputs, targets)

    statistics = {'move_FDA_up': move_FDA_up, 'move_FDA_down': move_FDA_down,
                  'cough_EDA_up': cough_EDA_up, 'cough_EDA_down': cough_EDA_down,
                  'snoring_EDA_up': snoring_EDA_up, 'snoring_EDA_down': snoring_EDA_down}

    return statistics

def evaluate_ensemble(model, data_loader):
    """Args:
    model: torch.nn.Module
    data_loader: torch.utils.data.DataLoader
    """

    device = next(model.parameters()).device

    for i, (waveform, target) in enumerate(data_loader):
        waveform, target = move_data_to_device(waveform, device), move_data_to_device(target, device)

        with torch.no_grad():
            model.eval()
            ensemble_output = model(waveform, None)

    # output: (batch_size, 10, 12)
    # target: (batch_size, 10, 8)

    cls_outputs = ensemble_output[:, :, [3, 4, 5, 9, 10, 11]].view(-1, 3).cpu()
    dur_outputs = ensemble_output[:, :, [1, 7]].view(-1, 1).cpu()

    cls_target = target[:, :, [1, 5]].view(-1).int().cpu()
    cls_targets = torch.eye(3)[cls_target].cpu()
    dur_targets = target[:, :, [3, 7]].view(-1).float().cpu()

    avg_precision = metrics.average_precision_score(cls_targets, cls_outputs, average=None)
    roc_auc = metrics.roc_auc_score(cls_targets, cls_outputs, average=None)

    dur_diff = torch.abs(dur_outputs - dur_targets)
    dur_score = torch.mean(dur_diff)

    statistics = {'average_precision': avg_precision, 'roc_auc': roc_auc, 'dur_score': dur_score}

    return statistics