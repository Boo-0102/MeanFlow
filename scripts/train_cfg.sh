#!/bin/bash
accelerate launch --num_processes=1 \
      train.py \
      --exp-name "cifar_exp4_cfg" \
      --output-dir "work_dir" \
      --resolution 32 \
      --learning-rate 0.00002 \
      --batch-size 24 \
      --gradient-accumulation-steps 2 \
      --checkpointing-steps 5000 \
      --warmup-steps 4000 \
      --allow-tf32 \
      --mixed-precision "bf16" \
      --epochs 100 \
      --path-type "linear" \
      --weighting "adaptive" \
      --time-sampler "logit_normal" \
      --time-mu -2.0 \
      --time-sigma 2.0 \
      --ratio-r-not-equal-t 0.75 \
      --adaptive-p 0.75 \
      --do-cfg True \
      --num-classes 10 \
      --cfg-drop-prob 0.1 \
      --cfg-omega 2.0 \
      --cfg-kappa 0.9
