import os
import torch
import argparse
import time
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
    print(colored(cfg, 'yellow'))
    print(colored('Your current save_path is {}!!!'.format(cfg['save_path']), 'blue'))

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(colored('You are using GPU[{}] for training!'.format(cfg['gpu']), 'green'))
    else:
        device = torch.device('cpu')
        print(colored('You are using CPU for training!', 'green'))

    augmentation = get_transformation(cfg)
    train_dataset, val_dataset, test_dataset = get_dataset(cfg, augmentation)

    train_sampler = CategoriesSampler(train_dataset.label_list, cfg['num_episodes_epoch'], cfg['way'], cfg['shot'] + cfg['query'])
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=32, pin_memory=True)
    
    val_sampler = CategoriesSampler(val_dataset.label_list, cfg['num_eval_episodes'], cfg['way'], cfg['shot'] + cfg['query'])
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=32, pin_memory=True)
    
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
    if cfg['init_weights'] == True:
        pth = torch.load(cfg['init_weights_path'])['params'] # original state_dict()
        pretrained_dict = OrderedDict()
        pretrained_dict = {k:v for k,v in pth.items() if 'fc' not in k}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # if cfg['model_type'] == 'ResNet12': # gpu
    model.encoder = torch.nn.DataParallel(model.encoder, device_ids=list(cfg['gpu']))
    
    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0
    
    timer = Timer()
    global_count = 0
    writer = SummaryWriter(logdir=cfg['save_path']) # should be changed to logdir in the latest version
        
    label = torch.arange(cfg['way'], dtype=torch.int8).repeat(cfg['query']).type(torch.LongTensor)
    if torch.cuda.is_available():
        label = label.cuda()
    
    for epoch in range(1, cfg['max_epoch'] + 1):
        model.train()
        lr_scheduler.step()
        tl = Averager()
        ta = Averager()
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, index_label = batch[0].cuda(), batch[1].cuda()
            else:
                data, index_label = batch[0], batch[1]
            p = cfg['shot'] * cfg['way']
            data_shot, data_query = data[:p], data[p:]

            rot_loss, MI_loss, fsl_loss, final_loss, acc_list = model(data_shot, data_query, 'train')
            
            ### update
            total_loss = cfg['ave_weight'] * (0.6*fsl_loss + 1.5*rot_loss + 10000*MI_loss) + cfg['final_weight'] * final_loss
            total_loss = total_loss / cfg['back_ward_step']
            total_loss.backward()

            writer.add_scalar('data/rot_loss', float(rot_loss), global_count)
            writer.add_scalar('data/MI_loss', float(MI_loss), global_count)
            writer.add_scalar('data/fsl_loss', float(fsl_loss), global_count)
            writer.add_scalar('data/final_loss', float(final_loss), global_count)
            writer.add_scalar('data/acc', float(acc_list[-1]), global_count)

            writer.add_scalar('data/total_loss', float(total_loss), global_count)

            print('epoch {}, train {}/{}, total_loss={:.4f}, final_loss={:.4f} acc={:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'
                  .format(epoch, i, len(train_loader), total_loss.item(), final_loss.item(), acc_list[0], acc_list[1], acc_list[2], acc_list[3], acc_list[4]))
            
            tl.add(total_loss.item())
            ta.add(acc_list[-1])

            # if (i+1) % cfg['back_ward_step'] == 0:
            optimizer.step()
            
            optimizer.zero_grad()


        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()
            
        print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = cfg['shot'] * cfg['way']
                data_shot, data_query = data[:p], data[p:]
                rot_loss, MI_loss, fsl_loss, final_loss, acc_list = model(data_shot, data_query)
                total_loss = cfg['ave_weight'] * (2*rot_loss + 0.5*MI_loss + 0.5*fsl_loss) + cfg['final_weight'] * final_loss

                vl.add(total_loss.item())
                va.add(acc_list[-1])

        vl = vl.item()
        va = va.item()
        writer.add_scalar('data/val_loss', float(vl), epoch)
        writer.add_scalar('data/val_acc', float(va), epoch)             
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va >= trlog['max_acc']:
            trlog['max_acc'] = va
            trlog['max_acc_epoch'] = epoch
            save_model('max_acc', cfg, model)          
                
        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, os.path.join(cfg['save_path'], 'trlog'))

        save_model('epoch-last', cfg, model)

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / cfg['max_epoch'])))
    
    writer.close()

        # lr_scheduler.step()

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FSRC_BASE')
    parser.add_argument('--config_env', default='./configs/env.yaml',
                        help='Config file for the environment')
    parser.add_argument('--config_exp', default='./configs/NWPU_train_val_test/resnet12.yaml',
                        help='Config file for the experiment')
    args = parser.parse_args()

    main()
