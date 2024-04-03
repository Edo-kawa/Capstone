import torch
from scipy import ndimage
from sklearn import metrics
from functions import (move_data_to_device, generate_framewise_target)

def framewise_misclassification_handling(framewise_outputs):
    temp = torch.argmax(framewise_outputs, dim=2).numpy()

    # Apply opening operator with diameter=5
    # Single or continuous event frames can be filtered out 
    # if the number of these continuous frames is less than the operator diameter
    opening_result = ndimage.grey_erosion(temp, size=5)
    opening_result = ndimage.grey_dilation(opening_result, size=5)

    # Apply closing operator with diameter=5
    # to connect those event areas with narrow gaps between them
    closing_result = ndimage.grey_dilation(opening_result, size=5)
    closing_result = ndimage.grey_erosion(closing_result, size=5)

    # Apply dilation operator with diameter=5
    # This will result in an expansion of 2 frames on both ends of the event frame sequences
    dilation_result = ndimage.grey_dilation(closing_result, size=5)

    return torch.from_numpy(dilation_result)

def compute_detection_accuracy(framewise_output, target):
    # Compute FDA for frames related to 'move'
    FDA_correct_num = int((framewise_output[target == 2] == 2).sum())
    FDA_target_num = int((target == 2).sum())
    move_FDA = FDA_correct_num / FDA_target_num

    # Compute EDA for frames related to 'snoring' and 'cough'
    pred_events = torch.unique_consecutive(framewise_output[framewise_output != 0])
    target_events = torch.unique_consecutive(target[target != 0])

    cough_pred_num = (pred_events == 1).sum()
    snoring_pred_num = (pred_events == 3).sum()

    cough_target_num = (target_events == 1).sum()
    snoring_target_num = (target_events == 3).sum()

    cough_EDA = min(cough_pred_num / cough_target_num, 1.)
    snoring_EDA = min(snoring_pred_num / snoring_target_num, 1.)

    return move_FDA, cough_EDA, snoring_EDA

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

    (batch_size, frames_num, classes_num) = framewise_outputs.shape

    smoothed_outputs = framewise_misclassification_handling(framewise_outputs)
    targets = torch.argmax(targets, dim=2)

    result = [0, 0, 0]

    for batch_id in range(batch_size):
        temp = compute_detection_accuracy(smoothed_outputs[batch_id], targets[batch_id])
        result[0] += temp[0]
        result[1] += temp[1]
        result[2] += temp[2]

    statistics = {'move_FDA': result[0]/batch_size, 'cough_EDA': result[1]/batch_size, 
                  'snoring_EDA': result[2]/batch_size}

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