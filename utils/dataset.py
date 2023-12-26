import numpy as np
import argparse
import csv
import os
import glob
import datetime
import time
import logging
import h5py
import librosa

from utilities import (create_folder, get_filename, create_logging, 
    float32_to_int16, pad_or_truncate)
import config


def download_wavs(args):
    """Download videos and extract audio in wav format.
    """

    # Paths
    csv_path = args.csv_path
    audios_dir = args.audios_dir
    mini_data = args.mini_data
    
    if mini_data:
        logs_dir = '_logs/download_dataset/{}'.format(get_filename(csv_path))
    else:
        logs_dir = '_logs/download_dataset_minidata/{}'.format(get_filename(csv_path))
    
    create_folder(audios_dir)
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Download log is saved to {}'.format(logs_dir))

    # Read csv
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    lines = lines[3:]   # Remove csv head info

    if mini_data:
        lines = lines[0 : 10]   # Download partial data for debug
    
    download_time = time.time()

    # Download
    for (n, line) in enumerate(lines):
        
        items = line.split(', ')
        audio_id = items[0]
        start_time = float(items[1])
        end_time = float(items[2])
        label = items[3].split('"')[1].split(',')
        duration = end_time - start_time

        logging.info('{} {} start_time: {:.1f}, end_time: {:.1f}'.format(
            n, audio_id, start_time, end_time))
        
        # Download full video of whatever format
        video_name = os.path.join(audios_dir, '_Y{}.%(ext)s'.format(audio_id))
        os.system("youtube-dl --quiet -o '{}' -x 'https://www.youtube.com/watch?v={}'"\
            .format(video_name, audio_id))

        video_paths = glob.glob(os.path.join(audios_dir, '_Y' + audio_id + '.*'))

        # If download successful
        if len(video_paths) > 0:
            video_path = video_paths[0]     # Choose one video

            # Add 'Y' to the head because some video ids are started with '-'
            # which will cause problem
            audio_path = os.path.join(audios_dir, 'Y' + audio_id + '.wav')

            # Extract audio in wav format
            os.system("ffmpeg -loglevel panic -i {} -ac 1 -ar 32000 -ss {} -t 00:00:{} {} "\
                .format(video_path, 
                str(datetime.timedelta(seconds=start_time)), duration, 
                audio_path))
            
            # Remove downloaded video
            os.system("rm {}".format(video_path))
            
            logging.info("Download and convert to {}".format(audio_path))

                
    logging.info('Download finished! Time spent: {:.3f} s'.format(
        time.time() - download_time))

    logging.info('Logs can be viewed in {}'.format(logs_dir))


def pack_waveforms_to_hdf5(args):
    """Pack waveform and target of several audio clips to a single hdf5 file. 
    This can speed up loading and training.
    """

    # Arguments & parameters
    audios_dir = args.audios_dir
    waveforms_hdf5_path = args.waveforms_hdf5_path
    mini_data = args.mini_data
    train = args.train

    clip_samples = config.clip_length
    labels = config.labels
    classes_num = config.classes_num
    sample_rate = config.sample_rate

    labels_num = len(labels)

    # Paths
    if mini_data:
        prefix = 'mini_'
        waveforms_hdf5_path += '.mini'
    else:
        prefix = ''

    create_folder(os.path.dirname(waveforms_hdf5_path))

    logs_dir = '_logs/pack_waveforms_to_hdf5/{}{}'.format(prefix, get_filename(audios_dir))
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Write logs to {}'.format(logs_dir))

    # Pack waveform to hdf5
    total_time = time.time()

    with h5py.File(waveforms_hdf5_path, 'w') as hf:
        audios_num = 90 if train else 30

        hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
        hf.create_dataset('waveform', shape=((audios_num, clip_samples)), dtype=np.int16)
        hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=np.bool_)
        hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)

        hf_ix = 0

        for ix, lb in enumerate(labels):
            class_dir = os.path.join(audios_dir, lb)
            audios_names = os.listdir(class_dir)

            # Pack waveform & target of several audio clips to a single hdf5 file
            for n in range(audios_num/3):
                audio_path = os.path.join(class_dir, audios_names[n])

                if os.path.isfile(audio_path):
                    logging.info('{} {}'.format(n, audio_path))
                    audio_name = lb + audios_names[n]
                    (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
                    target = np.zeros(labels_num, dtype=np.bool_)
                    target[ix] = 1

                    audio = pad_or_truncate(audio, clip_samples)

                    hf['audio_name'][hf_ix] = audio_name.encode()
                    hf['waveform'][hf_ix] = float32_to_int16(audio)
                    hf['target'][hf_ix] = target

                    hf_ix += 1
                else:
                    logging.info('{} File does not exist! {}'.format(n, audio_path))

    logging.info('Write to {}'.format(waveforms_hdf5_path))
    logging.info('Pack hdf5 time: {:.3f}'.format(time.time() - total_time))
          

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    # parser_split = subparsers.add_parser('split_unbalanced_csv_to_partial_csvs')
    # parser_split.add_argument('--unbalanced_csv', type=str, required=True, help='Path of unbalanced_csv file to read.')
    # parser_split.add_argument('--unbalanced_partial_csvs_dir', type=str, required=True, help='Directory to save out split unbalanced partial csv.')

    parser_download_wavs = subparsers.add_parser('download_wavs')
    parser_download_wavs.add_argument('--csv_path', type=str, required=True, help='Path of csv file containing audio info to be downloaded.')
    parser_download_wavs.add_argument('--audios_dir', type=str, required=True, help='Directory to save out downloaded audio.')
    parser_download_wavs.add_argument('--mini_data', action='store_true', default=False, help='Set true to only download 10 audios for debugging.')

    parser_pack_wavs = subparsers.add_parser('pack_waveforms_to_hdf5')
    # parser_pack_wavs.add_argument('--csv_path', type=str, required=True, help='Path of csv file containing audio info to be downloaded.')
    parser_pack_wavs.add_argument('--audios_dir', type=str, required=True, help='Directory to save out downloaded audio.')
    parser_pack_wavs.add_argument('--waveforms_hdf5_path', type=str, required=True, help='Path to save out packed hdf5.')
    parser_pack_wavs.add_argument('--mini_data', action='store_true', default=False, help='Set true to only download 10 audios for debugging.')
    parser_pack_wavs.add_argument('--train', action='store_true', default=True, help='Set true to pack training wavs.')

    args = parser.parse_args()
    
    if args.mode == 'download_wavs':
        download_wavs(args)

    elif args.mode == 'pack_waveforms_to_hdf5':
        pack_waveforms_to_hdf5(args)

    else:
        raise Exception('Incorrect arguments!')