import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
from sklearn.metrics import average_precision_score, roc_auc_score

from models import (iSleepEventDetector)
# from eval import evaluate
import config
from data import (read_ensemble_data)
from eval import (compute_detection_accuracy)

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

    (train_samples, eval_samples, test_samples) = read_ensemble_data(data_dir, sample_rate=sample_rate, model_type='DT')

    onehot_train_targets = np.asarray(train_samples['target'])
    train_targets = np.argmax(onehot_train_targets, axis=2)

    onehot_eval_targets = np.asarray(eval_samples['target'])
    eval_targets = np.argmax(onehot_eval_targets, axis=2)

    onehot_test_targets = np.asarray(test_samples['target'])
    test_targets = np.argmax(onehot_test_targets, axis=2)

    model.fit(train_samples['data'], train_targets)

    eval_preds = model.predict(eval_samples['data'])
    onehot_eval_preds = np.eye(4)[np.int32(eval_preds)].reshape(-1, 4)
    move_pred_num, move_target_num, cough_pred_num, cough_target_num, \
        snoring_pred_num, snoring_target_num = compute_detection_accuracy(eval_preds, eval_targets)
        
    eval_avg_precision = average_precision_score(onehot_eval_targets.reshape(-1, 4), onehot_eval_preds, average=None)
    eval_roc_auc = roc_auc_score(onehot_eval_targets.reshape(-1, 4), onehot_eval_preds, average=None)

    print(f'move score: {move_pred_num / move_target_num}')
    print(f'cough pred num: {cough_pred_num}, cough target num: {cough_target_num}, cough score: {cough_pred_num / cough_target_num}')
    print(f'snoring pred num: {snoring_pred_num}, snoring target num: {snoring_target_num}, snoring score: {snoring_pred_num / snoring_target_num}')
    print(f'average precision: {eval_avg_precision}, roc auc score: {eval_roc_auc}')
    
    model.fit(eval_samples['data'], eval_targets)

    test_preds = model.predict(test_samples['data'])
    onehot_test_preds = np.eye(4)[np.int32(test_preds)].reshape(-1, 4)
    move_pred_num, move_target_num, cough_pred_num, cough_target_num, \
        snoring_pred_num, snoring_target_num = compute_detection_accuracy(test_preds, test_targets)
        
    test_avg_precision = average_precision_score(onehot_test_targets.reshape(-1, 4), onehot_test_preds, average=None)
    test_roc_auc = roc_auc_score(onehot_test_targets.reshape(-1, 4), onehot_test_preds, average=None)

    print(f'move score: {move_pred_num / move_target_num}')
    print(f'cough pred num: {cough_pred_num}, cough target num: {cough_target_num}, cough score: {cough_pred_num / cough_target_num}')
    print(f'snoring pred num: {snoring_pred_num}, snoring target num: {snoring_target_num}, snoring score: {snoring_pred_num / snoring_target_num}')
    print(f'average precision: {test_avg_precision}, roc auc score: {test_roc_auc}')


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

