import os
import torch
import netron
import argparse
import time
from termcolor import colored
from utils.config import create_config
from torch.utils.data import DataLoader

from SSL.models.ssl_proto import SSL_boost 
from SSL.dataloader.samplers import CategoriesSampler
from utils.common_config import get_dataset, get_transformation
from tensorboardX import SummaryWriter
from collections import OrderedDict


def main():
    # Retrieve config file
    cfg = create_config(args.config_env, args.config_exp)
    print(colored(cfg, 'yellow'))

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']
        print(colored('You are using GPU[{}] for training!'.format(cfg['gpu']), 'green'))
    else:
        print(colored('You are using CPU for training!', 'green'))

    if cfg['dataset'] == 'MiniImageNet':
        from SSL.dataloader.mini_imagenet import MiniImageNet as Dataset        
    else:
        raise ValueError('Non-supported Dataset.')

    augmentation = get_transformation(cfg)
    train_dataset, val_dataset, test_datasetdataset = get_dataset(cfg, augmentation)

    train_sampler = CategoriesSampler(train_dataset.label_list, cfg['num_episodes_epoch'], cfg['way'], cfg['shot'] + cfg['query'])
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)
    
    val_sampler = CategoriesSampler(val_dataset.label_list, cfg['num_eval_episodes'], cfg['way'], cfg['shot'] + cfg['query'])
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    
    model = SSL_boost(cfg, dropout = cfg['dropout'])
    # 可视化网络模型
    # print(model)
    # x = torch.randn(16, 3, 40, 40)
    # modelData = "model_visual.pth"
    # torch.onnx.export(model, (x, x, 'train'), modelData)
    # netron.start(modelData)  # 输出网络结构
    param_list = [{'params': model.encoder.parameters(), 'lr': cfg['lr']},
                {'params': model.slf_attn.parameters(), 'lr': cfg['lr'] * cfg['lr_mul']},
                {'params': model.Rotation_classifier.parameters(), 'lr': cfg['lr']}]

    optimizer = torch.optim.SGD(param_list, lr=cfg['lr'], momentum=cfg['momentum'], nesterov=cfg['nesterov'], weight_decay=cfg['weight_decay']) # 0.9 True
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])        

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        model = model.cuda()

    # load pre-trained model (no FC weights)
    pth = torch.load(cfg['init_weights'])['params'] # original state_dict()
    pretrained_dict = OrderedDict()
    pretrained_dict = {k:v for k,v in pth.items() if 'fc' not in k}
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if cfg['model_type'] == 'ResNet12': # gpu
        model.encoder = torch.nn.DataParallel(model.encoder, device_ids=list(range(p['gpu'])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FSRC_BASE')
    parser.add_argument('--config_env', default='./configs/env.yaml',
                        help='Config file for the environment')
    parser.add_argument('--config_exp', default='./configs/mini_imagenet/conv64.yaml',
                        help='Config file for the experiment')
    args = parser.parse_args()

    main()