# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math

import torch

import rfdetr.util.misc as utils


def create_lr_scheduler(optimizer, args, dataset_train):
    """
    Create learning rate scheduler based on args configuration.

    Args:
        optimizer: Optimizer instance
        args: Arguments namespace with lr_scheduler, warmup_epochs, epochs, lr_drop, lr_min_factor
        dataset_train: Training dataset for calculating steps per epoch

    Returns:
        Learning rate scheduler
    """
    # for cosine annealing, calculate total training steps and warmup steps
    total_batch_size_for_lr = args.batch_size * utils.get_world_size() * args.grad_accum_steps
    num_training_steps_per_epoch_lr = (len(dataset_train) + total_batch_size_for_lr - 1) // total_batch_size_for_lr
    total_training_steps_lr = num_training_steps_per_epoch_lr * args.epochs
    warmup_steps_lr = num_training_steps_per_epoch_lr * args.warmup_epochs

    def lr_lambda(current_step: int):
        if current_step < warmup_steps_lr:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps_lr))
        else:
            # Cosine annealing from multiplier 1.0 down to lr_min_factor
            if args.lr_scheduler == 'cosine':
                progress = float(current_step - warmup_steps_lr) / float(max(1, total_training_steps_lr - warmup_steps_lr))
                return args.lr_min_factor + (1 - args.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
            elif args.lr_scheduler == 'step':
                if current_step < args.lr_drop * num_training_steps_per_epoch_lr:
                    return 1.0
                else:
                    return 0.1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return lr_scheduler

