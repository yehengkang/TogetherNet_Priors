import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nets.yolo import YoloBody
from nets.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch


def init_distributed(local_rank):
    """Initialize distributed training."""
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')


def create_dataloaders(train_dataset, val_dataset, batch_size, num_workers, distributed):
    """Build DataLoader with optional DistributedSampler."""
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler   = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    gen = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=yolo_dataset_collate,
    )
    gen_val = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=yolo_dataset_collate,
    )
    return gen, gen_val, train_sampler, val_sampler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TogetherNet DDP Training')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for DistributedDataParallel. '
                             'Use -1 (default) for single-GPU / CPU training.')
    args = parser.parse_args()
    local_rank = args.local_rank
    distributed = local_rank != -1

    if distributed:
        init_distributed(local_rank)

    world_size = dist.get_world_size() if distributed else 1
    is_main = (local_rank in [-1, 0])
    Cuda            = True and torch.cuda.is_available()
    if distributed:
        Cuda = True
    classes_path    = 'model_data/rtts_classes.txt'
    # model_path      = 'model_data/yolox_s.pth'                 # Pretrained weights for better performance (COCO or VOC）
    model_path = ''                                              # No pretrained weights
    input_shape     = [640, 640]
    phi             = 's'
    mosaic              = False

    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 16

    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 16

    Freeze_Train        = False
    

    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01

    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4

    lr_decay_type       = "cos"

    save_period         = 1

    num_workers         = 4

    train_annotation_path   = '2007_train_fog.txt'
    val_annotation_path     = '2007_val_fog.txt'
    clear_annotation_path = '2007_train.txt'
    val_clear_annotation_path = '2007_val.txt'


    class_names, num_classes = get_classes(classes_path)

    device = torch.device('cuda:{}'.format(local_rank) if distributed else ('cuda' if Cuda else 'cpu'))

    model = YoloBody(num_classes, phi)
    weights_init(model)
    if model_path != '':
        if is_main:
            print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    yolo_loss    = YOLOLoss(num_classes)
    loss_history = LossHistory("logs/", model, input_shape=input_shape) if is_main else None

    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    model_train = model.train()
    if Cuda:
        model_train = model_train.to(device)
        cudnn.benchmark = True
    if distributed:
        model_train = torch.nn.parallel.DistributedDataParallel(
            model_train, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
    elif Cuda:
        model_train = torch.nn.DataParallel(model_train)

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    with open(clear_annotation_path, encoding='utf-8') as f:
        clear_lines = f.readlines()
    with open(val_clear_annotation_path, encoding='utf-8') as f:
        val_clear_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 64
        total_batch_size = batch_size * world_size
        Init_lr_fit     = max(total_batch_size / nbs * Init_lr, 1e-4)
        Min_lr_fit      = max(total_batch_size / nbs * Min_lr, 1e-6)

        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset error!")

        train_dataset   = YoloDataset(train_lines, clear_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, mosaic=mosaic, train = True)
        val_dataset     = YoloDataset(val_lines, val_clear_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, mosaic=False, train = False)
        gen, gen_val, train_sampler, val_sampler = create_dataloaders(
            train_dataset, val_dataset, batch_size, num_workers, distributed)

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)
                if val_sampler is not None:
                    val_sampler.set_epoch(epoch)

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 64
                total_batch_size = batch_size * world_size
                Init_lr_fit     = max(total_batch_size / nbs * Init_lr, 1e-4)
                Min_lr_fit      = max(total_batch_size / nbs * Min_lr, 1e-6)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset error！")

                gen, gen_val, train_sampler, val_sampler = create_dataloaders(
                    train_dataset, val_dataset, batch_size, num_workers, distributed)

                UnFreeze_flag = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, save_period, local_rank=local_rank)
