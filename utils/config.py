import numpy as np

sample_rate = 32000
clip_length = sample_rate * 4     # Audio clips are 4-second
frames_num = 10 * 4

labels = ['cough', 'move', 'snoring']

classes_num = len(labels)

lb_to_ix = {label : i for i, label in enumerate(labels)}
ix_to_lb = {i : label for i, label in enumerate(labels)}

