#!/usr/bin/env bash
set -e

# Sanity overfit on tiny data
python tokenizer/train_tokenizer.py --input data/tiny.txt --vocab_size 50000 --model_prefix tokenizer/spm

python train.py \
  --data data/tiny.txt \  --sp_model tokenizer/spm.model \  --out_dir runs/gpt2mini_tiny \  --vocab_size 50000 \  --n_ctx 512 \  --d_model 256 \  --n_layer 12 \  --n_head 8 \  --batch_size 64 \  --grad_accum 8 \  --lr 3e-4 \  --max_steps 500 \  --bf16
