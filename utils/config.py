import os
import yaml
import easydict
from easydict import EasyDict
import errno

def create_config(config_file_env, config_file_exp):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
        root_dir = os.path.abspath(root_dir)
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
    for k, v in config.items():
        cfg[k] = v
    
    
    base_dir = os.path.join(root_dir, cfg['dataset'])
    cfg['base_dir'] = base_dir
    mkdir_if_missing(base_dir)

    if cfg['work_mode'] == 'train':
        save_path1 = '-'.join([cfg['model_type'], str(cfg['embed_size']),'SSL', str(cfg['shot']), str(cfg['way'])])
        save_path2 = '_'.join([str(cfg['step_size']), str(cfg['gamma']), str(cfg['lr']), str(cfg['temperature'])])
        training_params = '_'.join(['mom', str(cfg['momentum']), 'wd', str(cfg['weight_decay']), 'bsz' ,str(cfg['back_ward_step'])])
        save_path = os.path.join(base_dir, save_path1, save_path2 + training_params)
        cfg['save_path'] = save_path
        mkdir_if_missing(save_path)

    return cfg 

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise