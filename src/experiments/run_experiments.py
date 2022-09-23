import json
import os
import logging

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.growable_models import GrowableModel
from src.outer_bohb import get_optimizer_and_crit, evaluate_config
from src.utils.utils import load_cfg, load_logs


def get_baseline_config():
    baseline_config = {
        'learning_rate_init': 0.01,
        'batch_size': 100,
        'n_conv_layers': 3,
        'kernel_size': 3,
        'n_channels_conv_0': (16, 16),  # (initial, maximum) number of channels
        'n_channels_conv_1': (32, 32),
        'n_channels_conv_2': (64, 64),
        'global_avg_pooling': False,
        'use_BN': True,
        'dropout_rate': 0.,
        'n_fc_layers': 0,
        'device': 'cuda',
        'data_dir': 'FashionMNIST',
        'optimizer': 'Adam',
        'train_criterion': 'CrossEntropy',
        'trigger_threshold': 0.5,  # no effect in baseline
        'max_params': None,  # no effect in baseline
    }
    return baseline_config


def run_baseline():
    cfg = get_baseline_config()
    name = 'baseline'
    cfg['name'] = name  # name will be used to log the data
    with open(os.path.join('cfgs', f'{name}.json'), 'w') as f:
        json.dump(cfg, f)

    logging.info(f'Doing experiment: {name}')
    evaluate_config(cfg, 0, '', budget=10, nfolds=5)

def run_north_select():
    cfg = get_baseline_config()
    cfg['reproduce_paper'] = True
    thresholds = [0.9]
    for i in range(3):
        cfg[f'n_channels_conv_{i}'] = (16, 256)
    for t in thresholds:
        name = f'reproduce_gamma_{t*100:.0f}'
        cfg['name'] = name  # name will be used to log the data
        cfg['trigger_threshold'] = t
        with open(os.path.join('cfgs', f'{name}.json'), 'w') as f:
            json.dump(cfg, f)

        logging.info(f'Doing experiment: {name}')
        evaluate_config(cfg, 0, '', budget=10, nfolds=5)

def run_threshold_grid():
    cfg = get_baseline_config()
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(3):
        cfg[f'n_channels_conv_{i}'] = (8, 256)
    for t in thresholds:
        name = f'const_threshold_{t*100:.0f}'
        cfg['name'] = name  # name will be used to log the data
        cfg['trigger_threshold'] = t
        with open(os.path.join('cfgs', f'{name}.json'), 'w') as f:
            json.dump(cfg, f)

        logging.info(f'Doing experiment: {name}')
        evaluate_config(cfg, 0, '', budget=10, nfolds=5)

def test_config(cfg, model_cfg_list=None):
    experiment_name = cfg['name'] if 'name' in cfg else None

    lr = cfg['learning_rate_init'] if cfg['learning_rate_init'] else 0.01
    batch_size = cfg['batch_size'] if 'batch_size' in cfg else 100
    data_dir = cfg['data_dir'] if cfg['data_dir'] else 'FashionMNIST'
    device = cfg['device'] if cfg['device'] else 'cpu'

    # Device configuration
    torch.manual_seed(0)
    model_device = torch.device(device)

    input_shape = (1, 28, 28)
    pre_processing = transforms.Compose([transforms.ToTensor(), ])

    train_data = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=pre_processing)
    test_data = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=pre_processing)

    # returns the cross validation accuracy
    num_epochs = 10
    score = []
    log_list = []

    logging.info(f'evaluating config:\n{cfg}')
    for i in range(5):

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

        model_cfg = cfg if model_cfg_list is None else model_cfg_list[i]
        print(model_cfg)
        model = GrowableModel(model_cfg, input_shape=input_shape, num_classes=len(train_data.classes),
                              enable_logging=bool(experiment_name)).to(model_device)

        print(model)

        model_optimizer, train_criterion = get_optimizer_and_crit(cfg)
        optimizer = model_optimizer(model.parameters(), lr=lr)
        train_criterion = train_criterion().to(device)

        for epoch in range(num_epochs):
            logging.info('#' * 50)
            logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
            train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, model_device)
            logging.info('Train accuracy %f', train_score)

        test_score = model.eval_fn(test_loader, device)
        score.append(test_score)

        if experiment_name:
            model_log = model.retrieve_logs()
            model_log['val_acc'] = test_score
            log_list.append(model_log)
            with open(os.path.join('logs', f'{experiment_name}.json'), 'w') as f:
                json.dump(log_list, f)

    return np.mean(test_score)

def run_baseline_test():
    name = 'test_baseline'

    cfg = get_baseline_config()
    cfg['name'] = name

    with open(os.path.join('cfgs', f'{name}.json'), 'w') as f:
        json.dump(cfg, f)
    test_config(cfg)

def run_opt_baseline_test():
    name = 'test_baseline_opt_cfg'

    cfg = get_baseline_config()

    # copy all hyperparameters over
    original_cfg = load_cfg('baseline_opt_cfg')
    cfg['name'] = name
    cfg['n_conv_layers'] = original_cfg['n_conv_layers']
    cfg['n_fc_layers'] = original_cfg['n_fc_layers']
    for i in range(cfg['n_conv_layers']):
        c = original_cfg[f'n_channels_conv_{i}']
        cfg[f'n_channels_conv_{i}'] = (c, c)
    for i in range(cfg['n_fc_layers']):
        c = original_cfg[f'n_channels_fc_{i}']
        cfg[f'n_features_fc_{i}'] = (c, c)
    cfg['learning_rate_init'] = original_cfg['learning_rate_init']

    with open(os.path.join('cfgs', f'{name}.json'), 'w') as f:
        json.dump(cfg, f)
    test_config(cfg)

def run_opt_cfg_test():
    name = 'test_opt_cfg'

    cfg = load_cfg('opt_cfg')
    cfg['name'] = name

    with open(os.path.join('cfgs', f'{name}.json'), 'w') as f:
        json.dump(cfg, f)
    test_config(cfg)

def run_opt_cfg_static_test():
    name = 'test_opt_cfg_static'
    cfg = load_cfg('test_opt_cfg')
    cfg['name'] = name

    logs = load_logs('test_opt_cfg')
    model_cfg_list = []
    for run_i in range(5):
        model_cfg = cfg.copy()
        for i in range(cfg['n_conv_layers']):
            c = logs[f'conv{i}_out_channel'][run_i][-1]
            model_cfg[f'n_channels_conv_{i}'] = (c, c)
        for i in range(cfg['n_fc_layers']):
            c = logs[f'fc{i}_out_channel'][run_i][-1]
            model_cfg[f'n_features_fc_{i}'] = (c, c)
        model_cfg_list.append(model_cfg)
    with open(os.path.join('cfgs', f'{name}.json'), 'w') as f:
        json.dump(cfg, f)
    test_config(cfg, model_cfg_list=model_cfg_list)


if __name__ == '__main__':
    logger = logging.getLogger("Growing ConvNet experiment")
    logging.basicConfig(level=logging.INFO)

    # only training data
    run_baseline()
    run_north_select()
    run_threshold_grid()

    # final evaluation: training and test data
    run_baseline_test()
    run_opt_cfg_test()
    run_opt_cfg_static_test()
    run_opt_baseline_test()
