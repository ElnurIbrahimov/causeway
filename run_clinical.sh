#!/bin/bash
cd /causeway
python -u train_on_transformer.py --domain clinical --model mistralai/Mistral-7B-v0.3 --d_causal 64 --epochs 200 --num_samples 50000 --lr 3e-4 --warmup_epochs 10 --batch_size 128
