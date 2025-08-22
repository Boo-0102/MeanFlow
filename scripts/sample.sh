#!/bin/bash
torchrun --nproc_per_node=1 sample.py \
    --ckpt "./work_dir/cifar_exp4/checkpoints/0005000.pt" \
    --per-proc-batch-size 16 \
    --num-samples 16 \
    --sample-dir "./samples" \
    --num-steps 1 \
    