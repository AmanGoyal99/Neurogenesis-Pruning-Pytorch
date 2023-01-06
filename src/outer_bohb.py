"""
===========================
Optimization using BOHB
===========================
"""
import argparse
import json
import os
import logging

import numpy as np
import torch

import ConfigSpace as CS
from ConfigSpace import Configuration
# from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
from sklearn.model_selection import StratifiedKFold

# from smac.configspace import ConfigurationSpace
# from smac.facade.smac_mf_facade import SMAC4MF
# from smac.scenario.scenario import Scenario
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms

from growable_models import GrowableModel
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/fashion_mnist_experiment')

def get_optimizer_and_crit(cfg):
    """No choice. Focus on the essentials for now."""
    assert 'optimizer' not in cfg or cfg['optimizer'] == 'Adam'
    model_optimizer = torch.optim.Adam  # not AdamW as the baseline uses Adam

    assert 'criterion' not in cfg or cfg['criterion'] == 'CrossEntropy'
    train_criterion = torch.nn.CrossEntropyLoss

    return model_optimizer, train_criterion


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
def evaluate_config(cfg: Configuration, seed: int, instance: str, budget: float, nfolds=3):
    """
    Creates an instance of the torch_model and fits the given data on it.
    This is the function-call we try to optimize. Chosen values are stored in
    the configuration (cfg).

    :param cfg: Configuration (basically a dictionary)
        configuration chosen by smac
    :param seed: int or RandomState
        used to initialize the rf's random generator
    :param instance: str
        used to represent the instance to use (just a placeholder for this example)
    :param budget: float
        used to set max iterations for the MLP
    Returns
    -------
    val_accuracy cross validation accuracy
    """
    experiment_name = cfg['name'] if 'name' in cfg else None

    lr = cfg['learning_rate_init'] if cfg['learning_rate_init'] else 0.01
    batch_size = cfg['batch_size'] if cfg['batch_size'] else 100
    data_dir = cfg['data_dir'] if cfg['data_dir'] else 'FashionMNIST'
    device = cfg['device'] if cfg['device'] else 'cpu'

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    input_shape = (3, 32, 32)
    # input_shape = (1,28,28)
    pre_processing = transforms.Compose([transforms.ToTensor(), ])

    train_val = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=pre_processing
    )
    trainset = datasets.CIFAR10( #changed
        root=data_dir,
        train=True,
        download=True,
        transform=pre_processing
    )

    valset = datasets.CIFAR10( #changed
        root=data_dir,
        train=False,
        download=True,
        transform=pre_processing
    )

    # returns the cross validation accuracy
    cv = StratifiedKFold(n_splits=nfolds, random_state=42, shuffle=True)  # to make CV splits consistent
    num_epochs = int(np.ceil(budget))
    score = []
    log_list = []

    logging.info(f'evaluating config:\n{cfg}')

    for train_idx, valid_idx in cv.split(trainset, trainset.targets):
        train_data = Subset(trainset, train_idx)
        val_data = Subset(trainset, valid_idx)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    # train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False)

        model = GrowableModel(cfg,
                                    input_shape=input_shape,
                                    num_classes=10,
                                    enable_logging=bool(experiment_name)).to(model_device)

        print(model)

        model_optimizer, train_criterion = get_optimizer_and_crit(cfg)
        optimizer = model_optimizer(model.parameters(), lr=lr)
        train_criterion = train_criterion().to(device)

        for epoch in range(num_epochs):
            logging.info('#' * 50)
            logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
            train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, model_device,epoch)
            logging.info('Train accuracy %f', train_score)
            # if i%100==99:
            #     writer.add_scalar("training/learning_rate",optimizer.param_groups[0]['lr'],i+len(train_val)*(epoch))
            #     writer.add_scalar("training/loss",train_loss/100,i+len(train_val)*(epoch))

        val_score = model.eval_fn(val_loader, device,train_criterion)
        score.append(val_score)

        if experiment_name:
            model_log = model.retrieve_logs()
            model_log['val_acc'] = val_score
            log_list.append(model_log)
            with open(os.path.join('logs', f'{experiment_name}.json'), 'w') as f:
                json.dump(log_list, f)

        val_acc = 1 - np.mean(score)  # because minimize
        return val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JAHS')
    parser.add_argument('--data_dir', type=str, default='./FashionMNIST')
    parser.add_argument('--working_dir', default='./tmp', type=str,
                        help="directory where intermediate results are stored")
    parser.add_argument('--runtime', default=3600, type=int, help='Running time allocated to run the algorithm')
    parser.add_argument('--max_epochs', type=int, default=50, help='maximal number of epochs to train the network')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cpu', help='device to run the models')
    parser.add_argument('--paper', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    logger = logging.getLogger("Growing ConvNets")
    logging.basicConfig(level=logging.INFO)

    # cs = ConfigurationSpace()

    # # hyperparameters
    # n_conv_layer = UniformIntegerHyperparameter("n_conv_layers", 2, 4, default_value=3)
    # n_fc_layer = UniformIntegerHyperparameter("n_fc_layers", 0, 2, default_value=0)
    # learning_rate_init = UniformFloatHyperparameter('learning_rate_init', 0.0001, 0.05, log=True, default_value=0.01)
    # if args.paper:
    #     trigger_threshold = UniformFloatHyperparameter('trigger_threshold', 0.7, 0.99, default_value=0.8)
    #     cs.add_hyperparameters([n_conv_layer, n_fc_layer, trigger_threshold, learning_rate_init])
    # else:
    #     conv_trigger_threshold = UniformFloatHyperparameter('conv_trigger_threshold', 0.0, 1.0, default_value=0.8)
    #     fc_trigger_threshold = UniformFloatHyperparameter('fc_trigger_threshold', 0.0, 1.0, default_value=0.4)

    #     cs.add_hyperparameters([n_conv_layer, n_fc_layer, conv_trigger_threshold, fc_trigger_threshold, learning_rate_init])

    #     # conditions to restrict the hyperparameter space
    #     use_fc_trigger = CS.conditions.GreaterThanCondition(fc_trigger_threshold, n_fc_layer, 0)
    #     cs.add_conditions([use_fc_trigger])

    data_dir = args.data_dir
    runtime = args.runtime
    device = args.device
    max_epochs = args.max_epochs
    working_dir = args.working_dir
    seed = args.seed

    # cs.add_hyperparameters([
    #     Constant('device', device),
    #     Constant('data_dir', data_dir),
    #     Constant('max_params', 3e4),
    #     Constant('reproduce_paper', 1 if args.paper else 0),
    #     Constant('max_pool',1)
    # ])
    default_config = {
        'n_conv_layers': 8, #changed
        'n_fc_layers': 2, #changed
        'trigger_threshold':0.9,
        # 'conv_trigger_threshold': 0.5,
        # 'max_params': 3000,
        # 'n_channels_conv_0': (64, 128),  # (initial, maximum) number of channels
        # 'n_channels_conv_1': (128, 256),
        # 'n_channels_conv_2': (256, 512),
        # 'n_channels_conv_3': (256, 512),
        # 'n_channels_conv_4': (512, 1024),
        # 'n_channels_conv_5': (512, 1024),
        # 'n_channels_conv_6': (512, 1024),
        # 'n_channels_conv_7': (512, 1024),
        # 'n_features_fc_0': (512,2**12),
        # 'n_features_fc_1': (2**12, 2**14),
        'n_channels_conv_0': (16, 128),  # (initial, maximum) number of channels
        'n_channels_conv_1': (32, 256),
        'n_channels_conv_2': (64, 512),
        'n_channels_conv_3': (64, 512),
        'n_channels_conv_4': (128, 1024),
        'n_channels_conv_5': (128, 1024),
        'n_channels_conv_6': (128, 1024),
        'n_channels_conv_7': (128, 1024),
        'n_features_fc_0': (2**10,2**13),
        'n_features_fc_1': (2**10, 2**13),
        # 'n_features_fc_2': (10, 4096),
        'kernel_size': 3,
        'global_avg_pooling': False,
        'max_pool': True, #New addition
        'use_BN': False, #changed
        'dropout_rate': 0.,
        # 'max_params': None,
        'reproduce_paper': True,
        'learning_rate_init': 2.5e-5,
        'batch_size': 100,
        'data_dir': 'CIFAR10',
        'optimizer': 'Adam',
        'train_criterion': 'CrossEntropy',
        'device':'cuda'
    }
    cfg = default_config
    evaluate_config(cfg, 0, '', budget=100, nfolds=2)
    # # SMAC scenario object
    # scenario = Scenario({"run_obj": "quality",  # optimize quality (alternative to runtime)
    #                      "wallclock-limit": runtime,  # max duration to run the optimization (in seconds)
    #                      "cs": cs,  # configuration space
    #                      'output-dir': working_dir,  # working directory where intermediate results are stored
    #                      "deterministic": "True",
    #                      })

    # # hyperband, training epochs used as budget
    # intensifier_kwargs = {'initial_budget': 3, 'max_budget': max_epochs, 'eta': 3}
    # smac = SMAC4MF(scenario=scenario, rng=np.random.RandomState(seed),
    #                tae_runner=evaluate_config,
    #                intensifier_kwargs=intensifier_kwargs,
    #                initial_design_kwargs={'n_configs_x_params': 1,  # how many initial configs to sample per parameter
    #                                       'max_config_fracs': .2})

    # #with open('default_cfg.json', 'w') as f:
    # #    json.dump(cs.get_default_configuration().get_dictionary(), f)
    # #def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(),
    # #                                      instance='1', budget=max_epochs, seed=seed)[1]
    # #print("Default Value: %.4f" % def_value)

    # # Start optimization
    # try:  # try finally used to catch any interrupt
    #     incumbent = smac.optimize()
    # finally:
    #     incumbent = smac.solver.incumbent

    # inc_value = smac.get_tae_runner().run(config=incumbent, instance='1', budget=max_epochs, seed=seed)[1]
    # print("Optimized Value: %.4f" % inc_value)

    # # store your optimal configuration to disk
    # opt_config = incumbent.get_dictionary()
    # with open('cfgs/opt_cfg.json', 'w') as f:
    #     json.dump(opt_config, f)

