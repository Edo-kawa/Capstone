import torch
from sklearn import metrics
from functions import (move_data_to_device, generate_framewise_target)

def evaluate(model, data_loader):
    """Args:
    model: torch.nn.Module
    data_loader: torch.utils.data.DataLoader
    """

    device = next(model.parameters()).device

    for i, (waveform, target) in enumerate(data_loader):
        waveform, target = move_data_to_device(waveform, device), move_data_to_device(target, device)

        with torch.no_grad():
            model.eval()
            framewise_output = model(waveform, None)

    (batch_size, frames_num, classes_num) = framewise_output.shape

    # print(framewise_output.shape, target.shape)

    outputs = framewise_output.view(batch_size*frames_num, classes_num).cpu()
    targets = target.view(batch_size*frames_num, classes_num).cpu()

    avg_precision = metrics.average_precision_score(targets, outputs, average=None)
    roc_auc = metrics.roc_auc_score(targets, outputs, average=None)

    statistics = {'average_precision': avg_precision, 'roc_auc': roc_auc}

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

    outputs = ensemble_output[:, :, [3, 4, 5, 9, 10, 11]].view(-1, 3).cpu()

    target = target[:, :, [1, 5]].view(-1).int().cpu()
    targets = torch.eye(3)[target].cpu()

    avg_precision = metrics.average_precision_score(targets, outputs, average=None)
    roc_auc = metrics.roc_auc_score(targets, outputs, average=None)

    statistics = {'average_precision': avg_precision, 'roc_auc': roc_auc}

    return statistics