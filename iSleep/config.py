import numpy as np

clip_duration = 4     # Audio clips are 4-second
frames_num_per_sec = 10

labels = ['ambient_noise', 'cough', 'move', 'snoring']

classes_num = len(labels)
frames_num = clip_duration * frames_num_per_sec

lb_to_ix = {label : i for i, label in enumerate(labels)}
ix_to_lb = {i : label for i, label in enumerate(labels)}

