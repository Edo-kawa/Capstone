import numpy as np

# sample_rate = 32000

clip_duration = 4     # Audio clips are 4-second
ensemble_duration = 10
frames_num_per_sec = 10

labels = ['ambient_noise', 'cough', 'move', 'snoring']
ensemble_labels = labels[1:]

classes_num = len(labels)
ensemble_classes_num = classes_num - 1

frames_num = clip_duration * frames_num_per_sec
ensemble_frames_num = ensemble_duration * frames_num_per_sec

lb_to_ix = {label : i for i, label in enumerate(labels)}
ix_to_lb = {i : label for i, label in enumerate(labels)}

