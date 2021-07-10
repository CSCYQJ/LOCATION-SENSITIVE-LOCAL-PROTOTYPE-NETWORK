"""Experiment Configuration"""
import os
import re
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('Local_Prototype_Visceral')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    """Default configurations"""
    input_size = (512, 512)
    seed = 1234
    cuda_visable = '0, 1, 2, 3, 4, 5, 6, 7'
    gpu_id = 1
    mode = 'train' # 'train'


    if mode == 'train':
        dataset = 'Visceral'  
        n_steps = 10000
        label_sets = 0
        batch_size = 4
        lr_milestones = [800,1600,2400]
        ignore_label = 255
        print_interval = 50
        val_interval = 100

        model = {
            'n_grid': 8,
            'overlap':True,
            'vote':'max',
        }

        task = {
            'n_ways': 1,
            'n_shots': 1,
            'n_queries': 1,
        }

        optim = {
            'lr': 1e-3,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }
    else:
        raise ValueError('Wrong configuration for "mode" !')


    exp_str = '_'.join(
        [dataset,]
        + [key for key, value in model.items() if value]
        + [f'sets_{label_sets}', f'{task["n_ways"]}way_{task["n_shots"]}shot_[{mode}]'])


    path = {
        'log_dir': './runs',
        'Visceral':{'data_dir': '../data/Visceral',
               'data_split': 'trainaug',}
    }

@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
