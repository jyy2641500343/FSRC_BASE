import os
import math
from termcolor import colored
import numpy as np
import torch
import torch.nn as nn 
import torchvision.transforms as transforms
from functools import partial
import torchvision.models as torchvision_models
import torch.utils.data.distributed

from SSL.dataloader.dataset import ImageDataset, MiniImageNet


def get_model(cfg, pretrain_path=None):
    
    return None

def get_optimizer(p, model):

    if p['optimizer'] == 'lars':
        optimizer = optimizer.LARS(model.parameters(), p['lr'],
                                        weight_decay=p['weight_decay'],
                                        momentum=p['momentum'])
    elif p['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), p['lr'],
                                weight_decay=p['weight_decay'])

    return optimizer

def get_data_transformation(p):
    
    normalize = transforms.Normalize(mean=p['normalize_mean'],
                                     std=p['normalize_std'])
    augmentation1 = [
        transforms.RandomResizedCrop(p['image_size'], scale=(p['crop_min'], 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(p['image_size'], scale=(p['crop_min'], 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    return augmentation1, augmentation2

def get_dataset(cfg, augmentation):
    transform = transforms.Compose(augmentation)
    train_dataset = ImageDataset(root_path=cfg['train_dataset_path'], transform=transform, cfg=cfg)
    val_dataset = ImageDataset(root_path=cfg['val_dataset_path'], transform=transform, cfg=cfg)
    test_dataset = ImageDataset(root_path=cfg['test_dataset_path'], transform=transform, cfg=cfg)
    
    # train_dataset = MiniImageNet('train', cfg)
    # val_dataset = MiniImageNet('val', cfg)
    # test_dataset = MiniImageNet('test', cfg)

    return train_dataset, val_dataset, test_dataset

def get_transformation(cfg):
    if cfg['dataset'] == 'MiniImageNet':
        augmentation = [
            transforms.Resize(92),
            transforms.CenterCrop(cfg['image_size'])
        ]
    elif cfg['dataset'] == 'NWPU_train_val_test':
        augmentation = [
            transforms.Resize(cfg['image_size']+int(0*cfg['image_size'])),
            transforms.CenterCrop(cfg['image_size'])
        ]
    elif cfg['dataset'] == 'UCMerced_LandUse':
        augmentation = [
            transforms.Resize(cfg['image_size']+int(0*cfg['image_size'])),
            transforms.CenterCrop(cfg['image_size'])
        ]
    elif cfg['dataset'] == 'WHU_RS19':
        augmentation = [
            transforms.Resize(cfg['image_size']+int(0*cfg['image_size'])),
            transforms.CenterCrop(cfg['image_size'])
        ]
    
    return augmentation
