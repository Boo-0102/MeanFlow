#!/bin/bash
torchrun --nproc_per_node=1 evaluate.py \
    --ckpt "./work_dir/cifar_exp2/checkpoints/0039050.pt" \
    --per-proc-batch-size 128 \
    --num-fid-samples 50000 \
    --sample-dir "./fid_dir" \
    --compute-metrics \
    --num-steps 1\
    --fid-ref "train"