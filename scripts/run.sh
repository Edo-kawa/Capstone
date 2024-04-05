CUDA_VISIBLE_DEVICES=0 python3 iSleep/train.py train \
    --data_dir='./datasets/ensemble_data' \
    --sample_rate=32000 \
    --window_size=3200 \
    --hop_size=3200 \
    --mel_bins=64 \
    --fmin=50 \
    --fmax=14000 \
    --model_type='MLP' \
    --mixup \
    --batch_size=32 \
    --learning_rate=1e-3 \
    --num_iters=10000 \
    --ft_iters=5000