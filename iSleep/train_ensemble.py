import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from eval import evaluate_ensemble
import config
from data import (read_ensemble_data, EventDataSet)
from ensemble_detector import EnsembleLoss, DetectNet
from functions import (move_data_to_device, Mixup, do_mixup)

def train(args):
    """Args:
      data_dir: str
      window_size: int
      hop_size: int
      mel_bins: int
      fmin: int
      fmax: int
      model_type: str
      if_mixup: bool
      batch_size: int
      learning_rate: float
      num_iters: int
      if_resume: bool
    """

    data_dir = args.data_dir
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    if_mixup = args.mixup
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_iters = args.num_iters
    if_resume = args.resume

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    num_workers = 2
    classes_num = config.ensemble_classes_num
    criterion = EnsembleLoss(classes_num=classes_num, noobj_scale=0.5, coord_scale=0.5)

    model = DetectNet(thresh1=0.1, thresh2=0.5, sample_rate=sample_rate, window_size=window_size,
                      hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, classes_num=classes_num)
    
    model.to(device)
    
    (train_samples, eval_samples, test_samples) = read_ensemble_data(data_dir=data_dir, sample_rate=sample_rate, model_type=model_type)
    TrainSet = EventDataSet(sr=sample_rate, samples=train_samples)
    EvalSet = EventDataSet(sr=sample_rate, samples=eval_samples)
    TestSet = EventDataSet(sr=sample_rate, samples=test_samples)

    train_loader = torch.utils.data.DataLoader(dataset=TrainSet,
                                              batch_size=batch_size if not if_mixup else batch_size*2, 
                                              shuffle=True,
                                              num_workers=num_workers, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(dataset=EvalSet,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=TestSet,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, pin_memory=True)
    
    if if_mixup:
        mixup = Mixup(mixup_alpha=1., random_seed=42)

    if if_resume:
        checkpoint_path = f'./checkpoints/event_detectors/{model_type}.pth'

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        cur_iter = checkpoint['iteration']
    else:
        checkpoint = {}
        cur_iter = 0
    
    if cur_iter >= num_iters:
        return

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    for cur in range(cur_iter, num_iters):
        for i, (waveform, target) in enumerate(train_loader):
            waveform, target = move_data_to_device(waveform, device), move_data_to_device(target, device)

            if cur % 500 == 0 or cur == 0:
                eval_statistics = evaluate_ensemble(model, eval_loader)
                test_statistics = evaluate_ensemble(model, test_loader)

                checkpoint['eval_statistics'] = eval_statistics
                checkpoint['test_statistics'] = test_statistics

                print(f'cur_iter: {cur}, eval_mAP: {np.mean(eval_statistics["average_precision"])}, roc_auc: {eval_statistics["roc_auc"]}')
                print(f'cur_iter: {cur}, test_mAP: {np.mean(test_statistics["average_precision"])}, roc_auc: {test_statistics["roc_auc"]}')

            if cur % 1000 == 0:
                checkpoint['model'] = model.state_dict()
                checkpoint['iteration'] = cur
                checkpoint_path = f'./checkpoints/event_detectors/{model_type}.pth'
                torch.save(checkpoint, checkpoint_path)

            model.train()

            if if_mixup:
                mixup_lambda = mixup.get_lambda(waveform.shape[0])
                mixup_lambda = move_data_to_device(mixup_lambda, device)
                target = do_mixup(target, mixup_lambda)
            else:
                mixup_lambda = None

            ensemble_output = model(waveform, mixup_lambda)
            (batch_size, frames_num, classes_num) = ensemble_output.shape

            loss = criterion(ensemble_output, target)
            loss.backward()
            print(f'Iter: {cur}, Loss: {loss}')

            optimizer.step()
            optimizer.zero_grad()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--data_dir', type=str, required=True)
    parser_train.add_argument('--sample_rate', type=int, default=32000)
    parser_train.add_argument('--window_size', type=int, default=1024)
    parser_train.add_argument('--hop_size', type=int, default=320)
    parser_train.add_argument('--mel_bins', type=int, default=64)
    parser_train.add_argument('--fmin', type=int, default=50)
    parser_train.add_argument('--fmax', type=int, default=14000) 
    parser_train.add_argument('--model_type', type=str, required=True, default='DetectNet')
    parser_train.add_argument('--mixup', action='store_true', default=False)
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--learning_rate', type=float, default=1e-3)
    parser_train.add_argument('--num_iters', type=int, default=10000)
    parser_train.add_argument('--resume', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error!')