import pytest
from src.given.cnn import torchModel
from src.growable_models import GrowableModel, GrowableLinear, GrowableConv2d, GrowableBlock
from src.utils.utils import reproducible_seed

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import math
from typing import Optional, Dict, Type
import logging


def test_grow_math():
    conv = GrowableConv2d(3, 20, 1, 10, 3)
    b = GrowableBlock(conv, trigger_threshold=0.5, enable_logging=False)

    # preprocessing: batch of 2, 3 channels, 4x4 each
    a = torch.arange(2*3*4*4).reshape((2, 3, 4, 4))
    a_fn = b._flatten_and_normalize_act(a)
    #import ipdb; ipdb.set_trace()
    assert list(a_fn.shape) == [32, 3]
    assert a_fn.max() == pytest.approx(1/math.sqrt(2) * a.max())

    s = torch.tensor([5., 4., 3., 0., 0.])
    edim, erank = b._edim_and_erank(s)
    assert edim == 3 and erank == pytest.approx(2.9374925023243565)

    weight, bias = b._generate_random_weights(200)
    assert weight.shape[0] >= 200 and bias.shape[0] >= 200
    with torch.no_grad():
        assert weight.std() == pytest.approx(conv.current_weights().std())

    input = torch.rand((20, 1, 9, 9))
    # simulation should yield same results
    assert torch.allclose(b._simulate_weights(input, conv.current_weights(), conv.current_bias()), b(input))

    basis = torch.tensor([[1, 2, 3], [2, 8, 1], [0, 0, 0]], dtype=conv.weight.dtype)
    new_w = torch.tensor([[1, 2, 3, 4], [2, 8, 1, 3], [0, 1, 0, 6]], dtype=conv.weight.dtype)
    idx = b._least_representable(basis, new_w)
    assert idx[0] == 3 and idx[1] == 1


def simulate_growing_layer(growing, reference, in_c, out_c, kernel_shape=None):
    if kernel_shape is None:
        kernel_shape = []

    # just do forward pass for fun
    if kernel_shape:
        input = torch.rand((5, in_c, 32, 32))
    else:
        input = torch.rand((5, in_c))
    x = growing(input)
    x_reference = reference(input)

    # compare weights
    assert (reference.weight == growing.current_weights()).all() and (reference.bias == growing.current_bias()).all()

    # grow input channels
    growing.grow_in_neurons_by(10)
    if kernel_shape:
        grown_input = torch.rand((5, in_c+10, 32, 32))
    else:
        grown_input = torch.rand((5, in_c+10))
    grown_input[:, :in_c] = input
    x = growing(grown_input)

    # growing the input should not change the output
    assert (x == x_reference).all()

    # grow output by 7 with zeros
    new_w = torch.zeros([7, in_c+10] + kernel_shape)
    new_b = torch.zeros(7)
    growing.grow_out_neurons_by(7, new_w, new_b)
    x = growing(grown_input)

    # check output dim and values
    assert list(x.shape[:2]) == [5, out_c+7]
    assert (x[:, :out_c] == x_reference).all()
    assert not x[:, out_c:].any()  # true if all zero

    # grow output with random weights
    new_w = torch.rand([9, in_c+10] + kernel_shape)
    new_b = torch.rand(9)
    growing.grow_out_neurons_by(9, new_w, new_b)
    x = growing(grown_input)

    # check output dim and values
    assert list(x.shape[:2]) == [5, out_c+7+9]
    assert (x[:, :out_c] == x_reference).all()
    assert (x[:, out_c:]).any()

def test_growing_conv2d():
    in_c = 20
    out_c = 10
    k = 3

    # create growing and not growing conv2d
    reproducible_seed(42)
    reference = nn.Conv2d(in_c, out_c, 3)
    assert list(reference.weight.shape) == [out_c, in_c, k, k]
    reproducible_seed(42)
    growing = GrowableConv2d(in_c+21, out_c+30, in_c, out_c, k)
    assert list(growing.weight.shape) == [out_c+30, in_c+21, k, k]

    simulate_growing_layer(growing, reference, in_c, out_c, kernel_shape=[k, k])

def test_growing_linear():
    in_c = 20
    out_c = 10

    # create growing and not growing conv2d
    reproducible_seed(42)
    reference = nn.Linear(in_c, out_c)
    assert list(reference.weight.shape) == [out_c, in_c]
    reproducible_seed(42)
    growing = GrowableLinear(in_c+21, out_c+30, in_c, out_c, growth_multiplier=1)
    assert list(growing.weight.shape) == [out_c+30, in_c+21]

    simulate_growing_layer(growing, reference, in_c, out_c)


def run_model(model_class: Type[torch.nn.Module], cfg: Dict, data_dir: str, max_epochs=10,
              seed: int = 0, device: str = 'cuda'):
    reproducible_seed(seed)

    model_device = torch.device(device)

    img_width = 28
    img_height = 28
    input_shape = (1, img_width, img_height)

    pre_processing = transforms.Compose([transforms.ToTensor(), ])

    train_val = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=pre_processing)
    test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=pre_processing)

    lr = cfg['learning_rate_init']
    batch_size = cfg['batch_size']

    model = model_class(cfg, input_shape=input_shape, num_classes=len(train_val.classes)).to(model_device)

    logging.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_criterion = torch.nn.CrossEntropyLoss().to(model_device)

    train_loader = DataLoader(dataset=train_val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

    for epoch in range(max_epochs):
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, max_epochs))
        train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, model_device)
        logging.info('Train accuracy %f', train_score)

    test_score = model.eval_fn(test_loader, device)
    logging.info(f'Test accuracy {test_score}')
    return test_score

def test_equal_to_baseline():
    baseline_cfg = {
        'learning_rate_init': 0.01,
        'batch_size': 100,
        'n_conv_layers': 3,
        'kernel_size': 3,
        'n_channels_conv_0': 16,
        'n_channels_conv_1': 32,
        'n_channels_conv_2': 64,
        'global_avg_pooling': False,
        'use_BN': False,  # diff from actual baseline
        'dropout_rate': 0.,  # diff from actual baseline
        'n_fc_layers': 0
    }

    growable_cfg = baseline_cfg.copy()
    growable_cfg.update({
        'n_channels_conv_0': (16, 16),  # (initial, maximum) number of channels
        'n_channels_conv_1': (32, 32),
        'n_channels_conv_2': (64, 64),
        'trigger_threshold': 0.4,
    })

    growable_score = run_model(model_class=GrowableModel, cfg=growable_cfg, data_dir='./FashionMNIST', max_epochs=1)
    baseline_score = run_model(model_class=torchModel, cfg=baseline_cfg, data_dir='./FashionMNIST', max_epochs=1)

    assert growable_score == pytest.approx(baseline_score)

