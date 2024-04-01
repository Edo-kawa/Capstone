CUDA_VISIBLE_DEVICES=0 python3 iSleep/train_dt.py train \
    --data_dir='./datasets/ensemble_data' \
    --sample_rate=32000 \
    --window_size=3200 \
    --hop_size=3200 \