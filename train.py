import argparse
import json
import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import debugpy
import torch
import torchvision
import torchvision.transforms as T
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from meanflow import MeanFlow
from model.unet import SongUNet
from utils.logger import create_logger

"""
For sh debug
"""
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


def get_lr(step, learning_rate, warmup_steps):
    """
    Get learning rate with linear warmup
    """
    if warmup_steps > 0 and step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    else:
        return learning_rate


@torch.no_grad()
def update_ema(ema_model, model, decay=0.99995):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # 新的EMA权重 = 旧的EMA权重 * 衰减率 + 当前模型权重 * (1 - 衰减率)
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # Set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    torch.backends.cudnn.benchmark = True
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save args as a JSON file
        args_dict = vars(args)
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)

        # Accelerator init tensorboard 
        current_time_str = datetime.now().strftime('%m-%d_%H-%M-%S')
        project_name_with_date = f"{args.exp_name}_{current_time_str}"
        accelerator.init_trackers(project_name=project_name_with_date)

        # Create Logger
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
        
        # Log all args for reference
        logger.info("Training arguments:")
        for arg, value in sorted(args_dict.items()):
            logger.info(f"  {arg}: {value}")
            
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Init Model
    model = SongUNet(img_resolution=32,
            in_channels=3,
            out_channels=3,
            label_dim=args.num_classes if args.do_cfg else 0,
            label_dropout=args.cfg_drop_prob
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    # Create loss function with all MeanFlow parameters
    loss_fn = MeanFlow(
        path_type=args.path_type, 
        # Add MeanFlow specific parameters
        weighting=args.weighting,
        time_sampler=args.time_sampler,
        time_mu=args.time_mu,
        time_sigma=args.time_sigma,
        ratio_r_not_equal_t=args.ratio_r_not_equal_t,
        adaptive_p=args.adaptive_p,
        # CFG
        do_cfg = args.do_cfg,
        cfg_omega=args.cfg_omega,
        cfg_kappa=args.cfg_kappa,
        num_classes=args.num_classes
    
    )
    if accelerator.is_main_process:
        logger.info(f"UNet Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data/cifar",
        train=True,
        download=True,
        transform=T.Compose([T.RandomHorizontalFlip(),T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    )
    
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset, batch_size=local_batch_size, shuffle=True, drop_last=True, num_workers=8
    )

    if accelerator.is_main_process:
        logger.info(f"Train Dataset contains {len(train_dataset):,} images")
        if args.warmup_steps > 0:
            logger.info(f"Using linear warmup for {args.warmup_steps} steps")
    steps_per_epoch = len(train_dataloader) // accelerator.gradient_accumulation_steps
    args.max_train_steps = args.epochs * steps_per_epoch // accelerator.num_processes

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']
        del ckpt

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    ) 
    grad_norm = torch.tensor(0.).to(device)

    # Train
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Update learning rate with warmup
            current_lr = get_lr(global_step, args.learning_rate, args.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            with accelerator.accumulate(model):
                loss = loss_fn(model, images, labels)
                loss_mean = loss.mean() # 对一个batch的loss取均值
                loss = loss_mean
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model)
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0 or global_step >= args.max_train_steps:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                "lr": current_lr
            }
            progress_bar.set_postfix(**logs)

            # Log to tensorboard
            if accelerator.sync_gradients:
                accelerator.log(logs, step=global_step)

            # Log to file periodically
            if accelerator.is_main_process and global_step % 100 == 0:
                logger.info(f"Step {global_step}: loss = {logs['loss']:.4f}, grad_norm = {logs['grad_norm']:.4f}, lr = {logs['lr']:.6f}")

            if global_step >= args.max_train_steps:
                break
        
        # Log epoch completion
        if accelerator.is_main_process:
            logger.info(f"Completed epoch {epoch+1}/{args.epochs}")
            
        if global_step >= args.max_train_steps:
            break

    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training completed!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="MeanFlow Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # CFG
    parser.add_argument("--do-cfg", type=bool, default=False)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.1)
    parser.add_argument("--cfg-omega", type=float, default=2.0)
    parser.add_argument("--cfg-kappa", type=float, default=0.0)

    # dataset
    parser.add_argument("--resolution", type=int,default=32)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=4800)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--warmup-steps", type=int, default=10000, help="Number of warmup steps for learning rate")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    
    # seed
    parser.add_argument("--seed", type=int, default=42)
    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # basic loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--weighting", default="adaptive", type=str, choices=["uniform", "adaptive"], help="Loss weighting type")
    
    # MeanFlow specific parameters
    parser.add_argument("--time-sampler", type=str, default="logit_normal", choices=["uniform", "logit_normal"], 
                       help="Time sampling strategy")
    parser.add_argument("--time-mu", type=float, default=-2.0, help="Mean parameter for logit_normal distribution")
    parser.add_argument("--time-sigma", type=float, default=2.0, help="Std parameter for logit_normal distribution")
    parser.add_argument("--ratio-r-not-equal-t", type=float, default=0.75, help="Ratio of samples where r≠t")
    parser.add_argument("--adaptive-p", type=float, default=1.0, help="Power param for adaptive weighting")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)