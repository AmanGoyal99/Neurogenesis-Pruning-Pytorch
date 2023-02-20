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
import matplotlib.pyplot as plt
import time 
# CUDA_VISIBLE_DEVICES = 1
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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
def visualize(epochs,relative_layer_sizes,exp_name,xlabel,ylabel):
    fig = plt.figure()
    ax = plt.axes()
    x = np.array(epochs) 
    y = np.array(relative_layer_sizes)
    plt.plot(x,y)
    title = exp_name.split('_')[0]
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig('/home/amangoya/Neurogenesis-pytorch/Neurogenesis-Pruning-Pytorch/src/09.02/plots/10th_epoch/' + exp_name + '.png')

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
    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                        transform=transforms.Compose([ 
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                                transforms.Resize((32,32))
                            ]))

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

    # model = GrowableModel(cfg,
    #                                 input_shape=input_shape,
    #                                 num_classes=10,
    #                                 enable_logging=bool(experiment_name)).to(model_device)


    logging.info(f'evaluating config:\n{cfg}')
    dataset = datasets.CIFAR10('../data', train=True, download=True,
                     transform=transforms.Compose([ 
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                            transforms.Resize((32,32))
                        ]))
    # train_set, val_set = torch.utils.data.random_split(dataset, lengths=[int(0.9*len(dataset)), int(0.1*len(dataset))])
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True)

    # test_loader = torch.utils.data.DataLoader(
    # datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.1307,), (0.3081,)),
    #                         transforms.Resize((32,32))
    #                     ])),
    # batch_size=128, shuffle=True)

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
    conv1_relative_layer_size = [] 
    conv2_relative_layer_size = []
    conv3_relative_layer_size = [] 
    conv4_relative_layer_size = []
    conv5_relative_layer_size = []
    conv6_relative_layer_size = []
    conv7_relative_layer_size = []
    conv8_relative_layer_size = [] 
    fc1_relative_layer_size = []
    fc2_relative_layer_size = []
    # to_add_track = []
    # edim_track = [] 
    start_time = time.time()
    added_neurons_track = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
    edim_track = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
    track_growth = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
    standard_channel_list = [64, 128, 256, 256, 512, 512, 512,512]

    # added_neurons_track = []
    # edim_track = []
    for epoch in range(num_epochs):
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        train_score, train_loss, conv_channel_list, out_max_channel_list, fc_channel_list,edim_track,to_add_track,track_growth = model.train_fn(optimizer, train_criterion, train_loader, model_device,epoch,edim_track,added_neurons_track,track_growth,standard_channel_list)
        # for i in range(len(conv_channel_list)):
        #     if not added_neurons_track[i]:
        #         added_neurons_track[i].append(conv_channel_list[i])
        #     else:
        #         added_neurons_track[i].append(conv_channel_list[i] - added_neurons_track[i][-1])
        # print(conv_channel_list)
        # print(out_max_channel_list)
        # print(fc_channel_list)
        # divide_by_2 = [2]*len(out_max_channel_list)
        # out_max_channel_list = out_max_channel_list/divide_by_2
        # standard_channel_list = [64, 128, 256, 256, 512, 512, 512,512]
        # print(standard_channel_list)
        logging.info('Train accuracy %f', train_score)                
        conv1_relative_layer_size.append(conv_channel_list[0]/standard_channel_list[0])
        conv2_relative_layer_size.append(conv_channel_list[1]/standard_channel_list[1])
        conv3_relative_layer_size.append(conv_channel_list[2]/standard_channel_list[2])
        conv4_relative_layer_size.append(conv_channel_list[3]/standard_channel_list[3])
        conv5_relative_layer_size.append(conv_channel_list[4]/standard_channel_list[4])
        conv6_relative_layer_size.append(conv_channel_list[5]/standard_channel_list[5])
        conv7_relative_layer_size.append(conv_channel_list[6]/standard_channel_list[6])
        conv8_relative_layer_size.append(conv_channel_list[7]/standard_channel_list[7])
        fc1_relative_layer_size.append(fc_channel_list[0]/4096)
        fc2_relative_layer_size.append(fc_channel_list[1]/4096)
    print('conv1_relative_layer_size',conv1_relative_layer_size)
    print('conv2_relative_layer_size',conv2_relative_layer_size)
    print('conv3_relative_layer_size',conv3_relative_layer_size)
    print('conv4_relative_layer_size',conv4_relative_layer_size)
    print('conv5_relative_layer_size',conv5_relative_layer_size)
    print('conv6_relative_layer_size',conv6_relative_layer_size)
    print('conv7_relative_layer_size',conv7_relative_layer_size)
    print('conv8_relative_layer_size',conv8_relative_layer_size)
    print('fc1_relative_layer_size',fc1_relative_layer_size)
    print('fc2_relative_layer_size',fc1_relative_layer_size)
    # print('to_add_track',added_neurons_track)
    print('edim',edim_track)
    print('len of edim_track[0]',len(edim_track[0]))
    print('to_add',to_add_track)
    print('len of to_add_track[0]',len(to_add_track[0]))
    exp_name = 'trigger_threshold_0.9_required_10_svdals_0.01_graphs_testing_cifar10_50_epochs_2/'
    end_time = time.time()
    total_time = end_time - start_time
    print('Total training time in seconds:',total_time)
    print('Total training time in hrs:',total_time/3600)
    ###Visualize neuron generation####
    # exit(0)
    epochs = [i+1 for i in range(len(train_loader)*10)]
    visualize(epochs,track_growth[0],'Conv1_neuron_growth','iterations','Relative Layer Sizes')
    visualize(epochs,track_growth[1],'Conv2_neuron_growth','iterations', 'Relative Layer Sizes')
    visualize(epochs,track_growth[2],'Conv3_neuron_growth','iterations', 'Relative Layer Sizes')
    visualize(epochs,track_growth[3],'Conv4_neuron_growth','iterations','Relative Layer Sizes')
    visualize(epochs,track_growth[4],'Conv5_neuron_growth','iterations', 'Relative Layer Sizes')
    visualize(epochs,track_growth[5],'Conv6_neuron_growth','iterations', 'Relative Layer Sizes')
    visualize(epochs,track_growth[6],'Conv7_neuron_growth','iterations','Relative Layer Sizes')
    visualize(epochs,track_growth[7],'Conv8_neuron_growth','iterations', 'Relative Layer Sizes')

    ###effdim vs epochs####
    visualize(epochs,edim_track[0],'Conv1_effdim_track','iterations', 'effdim')
    visualize(epochs,edim_track[1],'Conv2_effdim_track','iterations', 'effdim')
    visualize(epochs,edim_track[2],'Conv3_effdim_track','iterations', 'effdim')
    visualize(epochs,edim_track[3],'Conv4_effdim_track','iterations', 'effdim')
    visualize(epochs,edim_track[4],'Conv5_effdim_track','iterations', 'effdim')
    visualize(epochs,edim_track[5],'Conv6_effdim_track','iterations', 'effdim')
    visualize(epochs,edim_track[6],'Conv7_effdim_track','iterations', 'effdim')
    visualize(epochs,edim_track[7],'Conv8_effdim_track','iterations', 'effdim')

    ###to_add vs epochs###
    visualize(epochs,to_add_track[0],'Conv1_to_add_track','iterations','to_add')
    visualize(epochs,to_add_track[1],'Conv2_to_add_track','iterations','to_add')
    visualize(epochs,to_add_track[2],'Conv3_to_add_track','iterations','to_add')
    visualize(epochs,to_add_track[3],'Conv4_to_add_track','iterations','to_add')
    visualize(epochs,to_add_track[4],'Conv5_to_add_track','iterations','to_add')
    visualize(epochs,to_add_track[5],'Conv6_to_add_track','iterations','to_add')
    visualize(epochs,to_add_track[6],'Conv7_to_add_track','iterations','to_add')
    visualize(epochs,to_add_track[7],'Conv8_to_add_track','iterations','to_add')


    ###effdim vs to_add####
    visualize(edim_track[0],to_add_track[0],'Conv1_effdim_to_add','effdim','to_add')
    visualize(edim_track[1],to_add_track[1],'Conv2_effdim_to_add','effdim','to_add')
    visualize(edim_track[2],to_add_track[2],'Conv3_effdim_to_add','effdim', 'to_add')
    visualize(edim_track[3],to_add_track[3],'Conv4_effdim_to_add','effdim','to_add')
    visualize(edim_track[4],to_add_track[4],'Conv5_effdim_to_add','effdim','to_add')
    visualize(edim_track[5],to_add_track[5],'Conv6_effdim_to_add','effdim', 'to_add')
    visualize(edim_track[6],to_add_track[6],'Conv7_effdim_to_add','effdim','to_add')
    visualize(edim_track[7],to_add_track[7],'Conv8_effdim_to_add','effdim','to_add')
    
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

    total_params = sum(param.numel() for param in model.parameters())
    print('Total Params of Model',total_params)
    trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable Params of Model',trainable_params)

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
        # 'n_channels_conv_0': (64 , 64),  # (initial, maximum) number of channels
        # 'n_channels_conv_1': (128, 128),
        # 'n_channels_conv_2': (256 , 256),
        # 'n_channels_conv_3': (256 , 256),
        # 'n_channels_conv_4': (512 , 512),
        # 'n_channels_conv_5': (512 , 512),
        # 'n_channels_conv_6': (512 , 512),
        # 'n_channels_conv_7': (512 , 512),
        # 'n_features_fc_0': (2**12,2**12),
        # 'n_features_fc_1': (2**12, 2**12),
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
        'learning_rate_init': 3e-4,
        # 'learning_rate_init': 0.0001,
        'batch_size': 256,
        # 'batch_size': 100,
        'data_dir': 'CIFAR10',
        'optimizer': 'Adam',
        'train_criterion': 'CrossEntropy',
        'device':'cuda'
    }
    cfg = default_config
    evaluate_config(cfg, 0, '', budget=2, nfolds=2)
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

