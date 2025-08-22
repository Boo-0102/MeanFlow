import argparse
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from meanflow_sampler import meanflow_sampler
from model.unet import SongUNet


def main(args):
    """
    Run sampling for unconditional CIFAR-10.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    model = SongUNet(
        img_resolution=32,
        in_channels=3,
        out_channels=3,
        label_dim=args.num_classes if args.do_cfg else 0,

    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=f'cuda:{device}', weights_only=False)
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create folder to save samples:
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "")
    folder_name = f"cifar10-unconditional-{ckpt_string_name}-" \
                  f"steps-{args.num_steps}-seed-{args.global_seed}"
    sample_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_dir}")
    dist.barrier()

    n = args.per_proc_batch_size # 每个GPU的batch size
    global_batch_size = n * dist.get_world_size() # 总batch size
    total_samples = int(math.ceil(args.num_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Using {args.num_steps}-step sampling")
    
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    
    for _ in pbar:
        # Sample noise at full resolution for CIFAR-10
        z = torch.randn(n, 3, 32, 32, device=device)

        # Sample images using MeanFlow :
        with torch.no_grad():
            samples = meanflow_sampler(
                model=model, 
                latents=z,
                num_steps=args.num_steps,
                do_cfg=args.do_cfg,
                num_classes=args.num_classes,
                class_label = args.c
            )
            
            # Convert to [0, 255] range
            samples = (samples + 1) / 2.0
            samples = torch.clamp(255.0 * samples, 0, 255)
            samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_dir}/{index:06d}.png")
        total += global_batch_size

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=42)
    # logging/saving:
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a MeanFlow checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # sampling
    parser.add_argument("--per-proc-batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=1, help="Number of sampling steps")

    # CFG
    parser.add_argument("--do-cfg", type=bool, default=False)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--c", type=int, default=2)
    args = parser.parse_args()
    
    main(args)