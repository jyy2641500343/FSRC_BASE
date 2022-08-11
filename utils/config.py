import os
import yaml
import easydict
from easydict import EasyDict
import errno

def create_config(config_file_env, config_file_exp):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
    for k, v in config.items():
        cfg[k] = v
    
    base_dir = os.path.join(root_dir, cfg['dataset'])
    mkdir_if_missing(base_dir)

    return cfg 

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise