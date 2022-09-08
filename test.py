import torch
import os
import numpy as np
import argparse

from termcolor import colored
from utils.config import create_config

from SSL.models.ssl_proto import SSL_boost



def main():
    # Retrieve config file
    cfg = create_config(args.config_env, args.config_exp)
    print(colored(cfg, 'yellow'))

    model = SSL_boost(cfg, dropout = cfg['dropout'])
    print(model.KDloss)









if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FSRC_BASE')
    parser.add_argument('--config_env', default='./configs/env.yaml',
                        help='Config file for the environment')
    parser.add_argument('--config_exp', default='./configs/mini_imagenet/resnet12.yaml',
                        help='Config file for the experiment')
    args = parser.parse_args()

    main()