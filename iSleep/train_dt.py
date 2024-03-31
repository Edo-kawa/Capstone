import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
from sklearn.metrics import average_precision_score, roc_auc_score

from models import (iSleepEventDetector)
# from eval import evaluate
import config
from data import (read_train_data)

def train(args):
    """Args:
      data_dir: str
      window_size: int
      hop_size: int
      if_mixup: bool
    """

    data_dir = args.data_dir
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size

    model = iSleepEventDetector(window_size=window_size, hop_size=hop_size)

    (train_samples, eval_samples, test_samples) = read_train_data(data_dir, sample_rate=sample_rate)

    onehot_train_targets = np.array(train_samples['target'])
    train_targets = np.argmax(onehot_train_targets, axis=2)

    onehot_eval_targets = np.array(eval_samples['target'])
    eval_targets = np.argmax(onehot_eval_targets, axis=2)

    onehot_test_targets = np.array(test_samples['target'])
    test_targets = np.argmax(onehot_test_targets, axis=2)

    model.fit(train_samples['data'], train_targets)

    eval_preds = model.predict(eval_samples['data'])

    onehot_eval_targets = onehot_eval_targets.reshape(-1, 2)
    eval_ap = average_precision_score(onehot_eval_targets, eval_preds, average=None)
    eval_auc = roc_auc_score(onehot_eval_targets, eval_preds, average=None)

    model.fit(eval_samples['data'], eval_targets)

    test_preds = model.predict(test_samples['data'])

    onehot_test_targets = onehot_test_targets.reshape(-1, 2)
    test_ap = average_precision_score(onehot_test_targets, test_preds, average=None)
    test_auc = roc_auc_score(onehot_test_targets, test_preds, average=None)

    eval_statistics = {'average_precision': eval_ap, 'roc_auc': eval_auc}
    test_statistics = {'average_precision': test_ap, 'roc_auc': test_auc}

    print(f'Eval AP: {eval_ap}, ROC-AUC: {eval_auc}')
    print(f'Test AP: {test_ap}, ROC-AUC: {test_auc}')

    return eval_statistics, test_statistics

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--data_dir', type=str, required=True)
    parser_train.add_argument('--sample_rate', type=int, default=32000)
    parser_train.add_argument('--window_size', type=int, default=3200)
    parser_train.add_argument('--hop_size', type=int, default=3200)
    
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error!')

