import os
import logging
import numpy as np
from scipy import stats
from math import ceil, floor
import datetime
import pickle
import config


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def get_sub_filepaths(folder):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
    return paths
    
    
def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.2
    x = np.clip(x, -1, 1)
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)

def process_data(x, clipwise_target, sample_rate, padder):
    """
    Designed to process non-ensemble data.
    Pad or truncate all audio and their labels to specific length.
    Args:
        x: np.array (data_length)
        clipwise_target: np.array (classes_num)
        padder: np.array (classes_num)
    """

    audio_length = config.clip_duration * sample_rate
    diff = abs(audio_length - len(x))
    left_to_pad_or_truncate = diff // 2
    right_to_pad_or_truncate = diff - left_to_pad_or_truncate

    # Standard audio length = actual audio length: no need to change
    if diff == 0:
        return (x, np.tile(clipwise_target, (config.frames_num, 1)))
    # Standard audio length < actual audio length: truncate
    elif len(x) > audio_length:
        return (x[left_to_pad_or_truncate : len(x) - right_to_pad_or_truncate], np.tile(clipwise_target, (config.frames_num, 1)))

    # Standard audio length > actual audio length: pad
    left_clip = np.random.normal(0, 1, left_to_pad_or_truncate)
    right_clip = np.random.normal(0, 1, right_to_pad_or_truncate)
    left_clip /= np.max(np.abs(left_clip))
    right_clip /= np.max(np.abs(right_clip))
    x = np.concatenate((left_clip, x, right_clip), axis=0)
    
    frame_length = sample_rate // config.frames_num_per_sec
    left_frames_num = left_to_pad_or_truncate // frame_length
    right_frames_num = right_to_pad_or_truncate // frame_length
    mid_frames_num = config.frames_num - left_frames_num - right_frames_num
    
    framewise_target = np.concatenate((np.tile(padder, (left_frames_num, 1)), 
                                    np.tile(clipwise_target, (mid_frames_num, 1)),
                                    np.tile(padder, (right_frames_num, 1))), axis=0)

    return (x, framewise_target)

def generate_framewise_target(labels):
    """
    Designed to process ensemble data for frame-wise detection networks.
    Args:
        labels: np.array (bboxes_num, class_id+start_time+end_time=3)
    """

    frames_num = config.ensemble_frames_num

    framewise_target = np.zeros((frames_num, config.classes_num))

    for label in labels:
        class_id = int(label[0])

        right_frame_id = ceil(label[2] * config.frames_num_per_sec) - 1         # label[2] != 0
        left_frame_id = floor(label[1] * config.frames_num_per_sec)             # label[1] != 10

        for frame_id in range(left_frame_id, right_frame_id+1):
            framewise_target[frame_id, class_id] = 1

    return framewise_target

def d_prime(auc):
    d_prime = stats.norm().ppf(auc) * np.sqrt(2.0)
    return d_prime

class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        """Contain statistics of different training iterations.
        """
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pkl'.format(
            os.path.splitext(self.statistics_path)[0], 
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'val': [], 'test': []}

    def append(self, iteration, statistics, data_type):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        logging.info('    Dump statistics to {}'.format(self.statistics_path))
        logging.info('    Dump statistics to {}'.format(self.backup_statistics_path))
        
    def load_state_dict(self, resume_iteration):
        self.statistics_dict = pickle.load(open(self.statistics_path, 'rb'))

        resume_statistics_dict = {'val': [], 'test': []}
        
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['iteration'] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)
                
        self.statistics_dict = resume_statistics_dict