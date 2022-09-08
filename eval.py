import os
import torch
import argparse
import time
from tqdm import tqdm
import numpy as np
from termcolor import colored
from utils.config import create_config
from torch.utils.data import DataLoader

from SSL.models.ssl_proto import SSL_boost 
from SSL.dataloader.samplers import CategoriesSampler
from utils.common_config import get_dataset, get_transformation
from SSL.utils import pprint, save_model, Averager, Timer, count_acc, euclidean_metric, compute_confidence_interval, cfg_from_yaml_file
from tensorboardX import SummaryWriter
from collections import OrderedDict



def main():
    # Retrieve config file
    cfg = create_config(args.config_env, args.config_exp)
    print(colored(vars(cfg), 'yellow'))
    print(colored('Your current evaluation model is {}!!!'.format(cfg['model_name']), 'blue'))

    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(colored('You are using GPU[{}] for evaluation!'.format(cfg['gpu']), 'green'))
    else:
        device = torch.device('cpu')
        print(colored('You are using CPU for evaluation!', 'green'))

    augmentation = get_transformation(cfg)
    train_dataset, val_dataset, test_dataset = get_dataset(cfg, augmentation)
    sampler = CategoriesSampler(test_dataset.label_list, cfg['test_episodes'], cfg['way'], cfg['shot'] + cfg['query'])
    loader = DataLoader(test_dataset, batch_sampler=sampler, num_workers=32, pin_memory=True)
    test_acc_record = np.zeros((cfg['test_episodes'],))

    model = SSL_boost(cfg, dropout = cfg['dropout'])
    model.encoder = torch.nn.DataParallel(model.encoder, device_ids=list(cfg['gpu']))

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        model = model.cuda()
    model_path = os.path.join(cfg['base_dir'], '{}-{}-{}-{}-{}'.format(cfg['model_type'], cfg['embed_size'], 'SSL', cfg['shot'], cfg['way']), cfg['model_name'], 'max_acc.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['params'])
    model.eval()

    ave_acc = Averager()
    tasks_acc = [Averager() for i in range(model.trans_num)]
    label = torch.arange(cfg['way']).repeat(cfg['query'])
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader, 1)):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = cfg['way'] * cfg['shot']
            data_shot, data_query = data[:k], data[k:]
            rot_loss, MI_loss, fsl_loss, final_loss, acc_list = model(data_shot, data_query)

            ave_acc.add(acc_list[-1])
            for j in range(model.trans_num):
                tasks_acc[j].add(acc_list[j])

            test_acc_record[i-1] = acc_list[-1]
            #print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc_list[-1] * 100))
        
    m, pm = compute_confidence_interval(test_acc_record)
    for i in range(model.trans_num):
        print('Rotation {} acc is {:.4f}'.format(90*i, tasks_acc[i].item()))
    print('Test Acc {:.4f} + {:.4f}'.format(m, pm))

    ave_acc = np.array(test_acc_record).mean() * 100 
    acc_std = np.array(test_acc_record).std() * 100
    ci95 = 1.96 * np.array(test_acc_record).std() / np.sqrt(float(len(np.array(test_acc_record)))) * 100

    print('evaluation: accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%'%(ave_acc, acc_std, ci95))














if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FSRC_BASE')
    parser.add_argument('--config_env', default='./configs/env.yaml',
                        help='Config file for the environment')
    parser.add_argument('--config_exp', default='./configs/NWPU_train_val_test/eval.yaml',
                        help='Config file for the experiment')
    args = parser.parse_args()

    main()
