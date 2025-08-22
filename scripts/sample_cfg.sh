#!/bin/bash
torchrun --nproc_per_node=1 sample.py \
    --ckpt "path/to/ckpt_cfg" \
    --per-proc-batch-size 16 \
    --num-samples 16 \
    --sample-dir "./samples" \
    --num-steps 1 \
    --do-cfg True \
    --num-classes 10 \
    --c 1
    