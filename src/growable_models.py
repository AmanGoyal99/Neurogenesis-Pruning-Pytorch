"""
===================================================
Growable layers and full growing ConvNet model
===================================================
"""
from typing import Optional
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import time
import math
from scipy.stats import entropy
import os
from utils.utils import AvgrageMeter, accuracy


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/fashion_mnist_experiment')

class GrowableBatchNorm1d(nn.BatchNorm1d):
    """
    Extends torch.nn.BatchNorm1d to support growing of input size.

    Reserves space for maximum size when initialized.
    Only track_running_stats=False is supported. track_running_stats=True is default in nn.BatchNorm2d.
    """
    def __init__(self, max_num_features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert not self.track_running_stats and "running_stats are not supported in growing version"
        assert self.momentum is not None and "momentum needed for growing version"
        self.max_num_features = max_num_features

        if self.affine:
            factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}

            self.weight = nn.parameter.Parameter(torch.ones(self.max_num_features, **factory_kwargs))
            self.bias = nn.parameter.Parameter(torch.zeros(self.max_num_features, **factory_kwargs))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Simplified version. Compared to the original BatchNorm implementation, it does not support tracking mean,
        therefore training=True always set.
        """
        assert input.dim() == 2 or input.dim() == 3
        return F.batch_norm(input, running_mean=None, running_var=None,
                            weight=self.weight[:self.num_features], bias=self.bias[:self.num_features],
                            training=True, momentum=self.momentum, eps=self.eps)

    def grow_num_features_by(self, n: int):
        assert self.num_features + n <= self.max_num_features
        self.num_features += n


class GrowableBatchNorm2d(nn.BatchNorm2d):
    """
    Extends torch.nn.BatchNorm2d to support growing of input size.

    Reserves space for maximum size when initialized.
    Only track_running_stats=False is supported. track_running_stats=True is default in nn.BatchNorm2d.
    """
    def __init__(self, max_num_features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert not self.track_running_stats and "running_stats are not supported in growing version"
        assert self.momentum is not None and "momentum needed for growing version"
        self.max_num_features = max_num_features

        if self.affine:
            factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}

            self.weight = nn.parameter.Parameter(torch.ones(self.max_num_features, **factory_kwargs))
            self.bias = nn.parameter.Parameter(torch.zeros(self.max_num_features, **factory_kwargs))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Simplified version. Compared to the original BatchNorm implementation, it does not support tracking mean,
        therefore training=True always set.
        """
        assert input.dim() == 4
        return F.batch_norm(input, running_mean=None, running_var=None,
                            weight=self.weight[:self.num_features], bias=self.bias[:self.num_features],
                            training=True, momentum=self.momentum, eps=self.eps)

    def grow_num_features_by(self, n: int):
        assert self.num_features + n <= self.max_num_features
        self.num_features += n


class GrowableMainLayer(ABC):
    """Common ground for GrowableLinear and GrowableConv2d. For details, see the children."""
    @abstractmethod
    def current_weights(self):
        pass
    @abstractmethod
    def current_bias(self):
        pass
    @abstractmethod
    def simulate_weights(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        pass
    @abstractmethod
    def grow_in_neurons_by(self, n: int):
        pass
    @abstractmethod
    def grow_out_neurons_by(self, n: int, weights: torch.Tensor, bias: Optional[torch.Tensor]):
        pass
    @abstractmethod
    def input_growable(self):
        pass
    @abstractmethod
    def output_growable(self):
        pass
    @abstractmethod
    def room_to_grow_out_neurons(self):
        pass
    @abstractmethod
    def num_neurons(self):
        pass


class GrowableLinear(GrowableMainLayer, nn.Linear):
    """
    Extends torch.nn.Linear to support growing of input size.

    Reserves space for maximum size when initialized.
    Growing the output is not implemented.
    """
    def __init__(self, max_in_features: int, max_out_features: int, *args,
                 growth_multiplier: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.growth_multiplier = growth_multiplier

        factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}

        with torch.no_grad():
            init_weights = self.weight.data
            self.weight = nn.parameter.Parameter(
                torch.zeros((self.max_out_features, self.max_in_features), **factory_kwargs))
            self.weight.data[:self.out_features, :self.in_features] = init_weights

            if self.bias is not None:
                init_bias = self.bias.data
                self.bias = nn.parameter.Parameter(
                    torch.zeros(self.max_out_features, **factory_kwargs))
                self.bias[:self.out_features] = init_bias

    def current_weights(self):
        """
        Returns what you would expect from the .weight parameter of nn.Linear.
        Here, .weight parameter holds the maximum-size weights, some of which may not be active yet.
        """
        return self.weight[:self.out_features, :self.in_features]

    def current_bias(self):
        """See docstring of current_weights."""
        return self.bias[:self.out_features] if self.bias is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.current_weights(), self.current_bias())

    def simulate_weights(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        """Same as forward, but with custom weights and bias. (used during selection of new neurons)"""
        return F.linear(input, weight, bias)

    def grow_in_neurons_by(self, n: int):
        """
        Weights for new input-features are zero-initialized, meaning they do not influence the output
        until further training.
        """
        n *= self.growth_multiplier
        assert self.in_features + n <= self.max_in_features
        self.in_features += n

    def grow_out_neurons_by(self, n: int, weights: torch.Tensor, bias: Optional[torch.Tensor]):
        """Adds n output channels using given weights and bias parameters."""
        assert self.out_features + n <= self.max_out_features
        with torch.no_grad():
            self.weight[self.out_features: self.out_features+n, :self.in_features] = weights
            if self.bias is not None:
                self.bias[self.out_features: self.out_features+n] = bias
        self.out_features += n

    def input_growable(self):
        return self.in_features < self.max_in_features

    def output_growable(self):
        return self.out_features < self.max_out_features

    def room_to_grow_out_neurons(self):
        return self.max_out_features - self.out_features

    def num_neurons(self):
        if not type(self.out_features)==int:
            return self.out_features.item()
        return self.out_features


class GrowableConv2d(GrowableMainLayer, nn.Conv2d):
    """
    Extends torch.nn.Conv2d to support growing of input and output channel sizes.

    Reserves space for maximum size when initialized.
    reset_parameters() is not supported. Just create a new object if you have to.
    """
    def __init__(self, max_in_channels: int, max_out_channels: int, *args, **kwargs) -> None:
        # we do not support the full range of Conv2d parameters
        if 'groups' in kwargs:
            assert kwargs['groups'] == 1

        # create member variables and initialize weights
        super().__init__(*args, **kwargs)

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.initial_in_channels = self.in_channels
        self.initial_out_channels = self.out_channels

        factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}

        with torch.no_grad():
            init_weights = self.weight.data
            self.weight = nn.parameter.Parameter(
                torch.zeros((self.max_out_channels, self.max_in_channels, *init_weights.shape[2:]), **factory_kwargs))
            self.weight.data[:self.out_channels, :self.in_channels] = init_weights

            if self.bias is not None:
                init_bias = self.bias.data
                self.bias = nn.parameter.Parameter(
                    torch.zeros(self.max_out_channels, **factory_kwargs))
                self.bias[:self.out_channels] = init_bias

    def current_weights(self):
        """
        Returns what you would expect from the .weight parameter of Conv2d.
        Here, .weight parameter holds the maximum-size weights, some of which may not be active yet.
        """
        return self.weight[:self.out_channels, :self.in_channels]

    def current_bias(self):
        """See docstring of current_weights."""
        return self.bias[:self.out_channels] if self.bias is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super()._conv_forward(input, self.current_weights(), self.current_bias())

    def simulate_weights(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        """Same as forward, but with custom weights and bias. (used during selection of new neurons)"""
        return super()._conv_forward(input, weight, bias)

    def grow_in_neurons_by(self, n: int):
        """Input channels are zero-initialized, meaning they do not influence the output until further training."""
        assert self.in_channels + n <= self.max_in_channels
        self.in_channels += n

    def grow_out_neurons_by(self, n: int, weights: torch.Tensor, bias: Optional[torch.Tensor]):
        """Adds n output channels using given weights and bias parameters."""
        assert self.out_channels + n <= self.max_out_channels
        with torch.no_grad():
            self.weight[self.out_channels: self.out_channels+n, :self.in_channels] = weights
            if self.bias is not None:
                self.bias[self.out_channels: self.out_channels+n] = bias
        self.out_channels += n

    def input_growable(self):
        return self.in_channels < self.max_in_channels

    def output_growable(self):
        return self.out_channels < self.max_out_channels

    def room_to_grow_out_neurons(self):
        return self.max_out_channels - self.out_channels

    def num_neurons(self):
        if not type(self.out_channels)==int:
            return self.out_channels.item()
        return self.out_channels

class GrowableBlock(nn.Sequential):
    """
    Represents up to a full layer including non-linearity or pooling.

    Houses computing the effective rank for triggers and initializing new weights for orthogonal activations.
    Ingredients:
        - required: one main growable layer, dense or convolutional
        - optional:
            - growable batch-norm layer(s) that fits size of main growable layer
            - any non-linearity or pooling layer(s)
    """
    def __init__(self, *args, trigger_threshold, enable_logging, reproduce_paper=False, **kwargs):
        """
        Create full layer (including NL, pooling and BN) which supports growing.
        Inherits from nn.Sequential.

        :param args: layers for Sequential model, including one main growable layer
        :param trigger_threshold: threshold limiting growth
        :param enable_logging: indicates whether to create logs
        :param reproduce_paper: indicates whether to reproduce NORTH-Select
        :param kwargs: additional kwargs for Sequential model
        """
        super().__init__(*args, **kwargs)
        self.trigger_threshold = trigger_threshold
        self.activations = None
        self.prepare_growth = False

        self.paper_trigger = reproduce_paper
        self.trigger_holdoff = True
        self.rel_erank_mavg = trigger_threshold
        self.initial_edim = None  # only used with paper_trigger
        self.edim_log = []
        self.erank_log = []
        self.rel_erank_mavg_log = []
        self.out_channel_log = []
        self.logging = enable_logging

        main_growables = [module for module in self if isinstance(module, GrowableMainLayer)]
        assert len(main_growables) == 1 and "a block only holds one conv/dense layer"
        self.main_growable = [i for i, m in enumerate(self) if isinstance(m, GrowableMainLayer)][0]

    def _least_representable(self, a, b, a_svd=None):
        """
        Returns an index. index[0] specifies the column of b where most information is lost
        when projecting to the space spanned by columns of a, and so on.
        :param a: matrix a
        :param b: matrix b
        :param a_svd: singular value decomposition of a (not computing this twice saves resources)
        :return: index vector, where index[0] specifies the column of b where most information is lost
        """
        if a_svd is None:
            a_svd = torch.linalg.svd(a, full_matrices=False)
        U, svdvals, Vh = a_svd
        # U = torch.from_numpy(U).to('cuda') #required when np is used for svd calc
        # svdvals = torch.from_numpy(svdvals).to('cuda') # required when np is used for svd calc
        # Vh = torch.from_numpy(Vh).to('cuda') # required when np is used for svd calc
        # invert S (of U @ S @ V^T)
        mask = svdvals > torch.finfo(torch.float32).eps
        s_inv = svdvals
        s_inv[mask] = 1 / s_inv[mask]

        # invert a using A^inv = V @ S^inv @ U^T
        a_inv = Vh.t() * s_inv @ U.t()

        # e.g., (1000x64) x (64x1000) x (1000x102) with 102 trial-channels and 64 existing ones
        b_restored = a @ (a_inv @ b)
        #ainal_normed = (b / b.norm(dim=0)).nan_to_num()
        #a_restored_normed = (restored_activations / restored_activations.norm(dim=0)).nan_to_num()
        b_error = b - b_restored
        b_sse = torch.sum(b_error * b_error, dim=0)
        least_representable_idx = b_sse.argsort(descending=True)
        return least_representable_idx

    def _edim_and_erank(self, svdvals, log_results=False):
        """Compute epsilon-numerical rank and effective rank.
        :param svdvals: array of singular values
        :param log_results: whether to store the results
        :return: epsilon-numerical rank (edim) and effective rank (erank)
        """
        # edim = torch.sum(svdvals > 0.01) #changed from 0.1
        # svdvals = torch.from_numpy(svdvals) #required when using numpy for svd calc
        edim = torch.count_nonzero(svdvals>0.01) #changed
        normalized_svdvals = svdvals / torch.sum(svdvals)
        # print('sum_svdvals',torch.sum(svdvals))
        erank = torch.e ** entropy(normalized_svdvals.detach().cpu())

        if log_results:
            self.erank_log.append(erank.item())
            self.edim_log.append(edim.item())
        return edim,erank

    def _flatten_and_normalize_act(self, act):
        """Flattens the given matrix to 2d: (batch, w, h) x channel, and normalizes over batch."""
        x = 1./math.sqrt(act.shape[0]) * act.swapaxes(1, -1).flatten(end_dim=-2)
        # if len(act.shape) > 2:
        #     act = torch.transpose(torch.transpose(act, 0, 1).reshape(act.shape[1], -1), 0, 1)
        # y =  act.clone() / act.shape[1]**0.5
        return x

    def _generate_random_weights(self, n_required):
        """Generate (normalized uniform-random) weight and bias tensors, more neurons than required."""
        # trial_channels = 100 + n_required
        trial_channels = 10 + n_required  #changed
        conv = self[self.main_growable]
        current_weights = conv.current_weights()
        weight_size = [trial_channels] + list(current_weights.size()[1:])
        weights = torch.rand(weight_size, device=current_weights.device) - 0.5
        weights = weights / weights.std() * current_weights.std()
        if conv.bias is not None:
            bias = torch.rand(trial_channels, device=current_weights.device) - 0.5
            bias = bias / bias.std() * conv.current_bias().std()
        else:
            bias = None
        return weights, bias

    def _simulate_weights(self, input, weights, bias):
        """Simulate forward pass, computing activations that result from new weights for the main growable layer."""
        x = input
        for i, module in enumerate(self):
            if i == self.main_growable:
                x = module.simulate_weights(x, weights, bias)
            elif isinstance(module, (GrowableBatchNorm1d, GrowableBatchNorm2d)):
                # mean=0, var=1 batch norm is the default
                x = F.batch_norm(x, running_mean=None, running_var=None,
                                    weight=None, bias=None,
                                    training=True, momentum=module.momentum, eps=module.eps)
            else:
                x = module(x)
        return x

    def _compute_trigger(self, edim, erank):
        """Returns the number of neurons to add. ('trigger' is terminology from NORTH*)
        :param edim: eps.-numerical rank of activations (used in NORTH-Select)
        :param erank: effective rank, replaces edim in final approach
        :return: Number of neurons to add to this layer.
        """
        conv = self[self.main_growable]

        # rel_erank = erank / conv.num_neurons()
        # mavg_rate = 0.1
        # self.rel_erank_mavg = (1-mavg_rate) * self.rel_erank_mavg + mavg_rate * rel_erank
        # self.rel_erank_mavg_log.append(self.rel_erank_mavg)

        # the trigger function employed by Maile et al.
        if self.paper_trigger:
            rel_edim = edim / conv.num_neurons()
            # rel_edim = rel_erank  # CUSTOM WORKAROUND because eps-numerical rank does not work for these small networks
            # in the paper, they did not use trigger holdoff or smoothing of the epsilon-numerical rank
            if not self.initial_edim:
                self.initial_edim = rel_edim
            # print('rel_edim',rel_edim)
            to_add = max(0,math.floor(conv.num_neurons() * (rel_edim - self.trigger_threshold * self.initial_edim)))
            # to_add = max(0,edim-int(0.95*conv.num_neurons())) #changed
            # print(to_add)
            # x = edim-int(0.95*conv.num_neurons())
            # print(conv.num_neurons() * (rel_edim - self.trigger_threshold * self.initial_edim))
            to_add = min(to_add, conv.room_to_grow_out_neurons())
            # to_add = 0
            # print('room',conv.room_to_grow_out_neurons())
            # print('M',conv.num_neurons())
            # print((rel_edim - self.trigger_threshold * self.initial_edim))
            # print('FLoored',math.floor(conv.num_neurons() * (rel_edim - self.trigger_threshold * self.initial_edim)))
            # update threshold (adjustment to NORTH-Select trigger for VGG and WRN by the original authors)
            # otherwise layers grow to max size immediately
            self.initial_edim = max(self.initial_edim, rel_edim)
            return to_add

        if self.trigger_holdoff:
            if len(self.rel_erank_mavg_log) > 3 and (self.rel_erank_mavg >= np.array(self.rel_erank_mavg_log[-4:])).all():
                self.trigger_holdoff = False
            else:
                return 0

        rel_threshold = self.trigger_threshold
        to_add = max(0, math.ceil((self.rel_erank_mavg - rel_threshold) * conv.num_neurons()))
        to_add = min(to_add, conv.room_to_grow_out_neurons())

        if to_add > 0:
            self.trigger_holdoff = True
            # self.rel_erank_mavg = self.trigger_threshold

        return to_add

        #if self.initial_edim is None:
        #    if len(self.erank_log) < 4 or not (np.array(self.erank_log[-4:]) <= erank).all():
        #        return 0
        #    else:
        #        self.initial_edim = edim

        ## edim_metric = erank / flat_activations.shape[0]
        ## to_add = max(0, math.floor(edim - self.edim_threshold * self.initial_edim))
        #rel_threshold = self.trigger_threshold
        #to_add = max(0, math.ceil(erank - rel_threshold * conv.num_neurons()))
        ## rel_erank = erank / conv.out_channels
        ## to_add = (rel_erank - 0.5)  * conv.out_channels
        #to_add = min(to_add, conv.room_to_grow_out_neurons())
        #return to_add

    def grow(self, input_activations, output_activations, max_growth=None, only_logging=False):
        """
        Grow this layer (during training!), depending on outgoing activations.
        Input-activations are used when selecting new neurons.
        Uses the 'trigger' and 'initialization' model laid out in the NORTH* paper.

        :param input_activations: former input to this layer during forward pass
        :param output_activations: former output of this layer during forward pass
        :param max_growth: maximum number of neurons to add
        :param only_logging: indicates to not grow, but still compute metrics used for growing (for inspection)
        :return: number of neurons added
        """
        conv = self[self.main_growable]
        if self.logging:
            self.out_channel_log.append(conv.num_neurons())

        # flatten to ((batches*height*width) x channels)
        flat_activations = self._flatten_and_normalize_act(output_activations)
        # svdvals = torch.linalg.svdvals(1/math.sqrt(flat_activations.shape[1]) * flat_activations)
        # print(output_activations)
        # compute svd of flat_activations: A = U @ S @ V^T = svd( 1/sqrt(#samples) H )
        U, svdvals, Vh = torch.linalg.svd(flat_activations, full_matrices=False)
        # print('flat_activations',flat_activations)
        edim,erank = self._edim_and_erank(svdvals, log_results=self.logging)
        # print('edim',edim)
        to_add = self._compute_trigger(edim,erank)
        if max_growth is not None:
            to_add = min(to_add, max_growth)

        # for debugging purposes
        if only_logging:
            return 0

        if to_add > 0:
            # print('Has something to add')
            # generates more weights than required. we will then evaluate them and choose the best ones
            weights, bias = self._generate_random_weights(n_required=to_add)
            # trainable_params = sum(p.numel() for p in weights.parameters() if p.requires_grad)
            # print(trainable_params)
            # do a forward pass using the new weights
            new_activations = self._simulate_weights(input_activations, weights, bias)
            flat_new_activations = self._flatten_and_normalize_act(new_activations)

            # choose the weight from channels which cannot be represented in basis of columns in flat_activations
            weights_idx = self._least_representable(flat_activations, flat_new_activations, a_svd=(U, svdvals, Vh))[:to_add] 

            conv.grow_out_neurons_by(to_add, weights[weights_idx], bias[weights_idx])
            for module in list(self)[self.main_growable+1:]:
                if isinstance(module, (GrowableBatchNorm1d, GrowableBatchNorm2d)):
                    module.grow_num_features_by(to_add)
        # else:
        #     # print('Nothin to add')
        return to_add,edim

    def grow_input_by(self, n):
        for i, module in enumerate(self):
            if isinstance(module, (GrowableBatchNorm1d, GrowableBatchNorm2d)):
                module.grow_num_features_by(n)
            elif i == self.main_growable:
                module.grow_in_neurons_by(n)
                break

    def retrieve_logs(self):
        return {'edim': self.edim_log, 'erank': self.erank_log, 'out_channel': self.out_channel_log,
                'rel_erank_mavg': self.rel_erank_mavg_log}

    def num_neurons(self):
        return self[self.main_growable].num_neurons()

    def num_params(self):
        """Returns number of active parameters in main growable layer (dense or conv)."""
        l = self[self.main_growable]
        return np.prod(l.current_weights().shape) + np.prod(l.current_bias().shape)


class GrowableModel(nn.Module):
    """
    The model to optimize
    """

    default_config = {
        'n_conv_layers': 8, #changed
        'n_fc_layers': 3, #changed
        'trigger_threshold': 0.5,
        'max_params': None,
        # 'n_channels_conv_0': (64, 128),  # (initial, maximum) number of channels
        # 'n_channels_conv_1': (128, 256),
        # 'n_channels_conv_2': (256, 512),
        # 'n_channels_conv_3': (256, 512),
        # 'n_channels_conv_4': (512, 1024),
        # 'n_channels_conv_5': (512, 1024),
        # 'n_channels_conv_6': (512, 1024),
        # 'n_channels_conv_7': (512, 1024),
        # 'n_features_fc_0': (2**12,2**13),
        # 'n_features_fc_1': (2**12, 2**13),
        # 'n_features_fc_2': (10, 10),
        'kernel_size': 3,
        'global_avg_pooling': False,
        'max_pool': True, #New addition
        'use_BN': False, #changed
        'dropout_rate': 0.,
        'max_params': None,
        'reproduce_paper': False,
    }

    def __init__(self, config, input_shape=(1, 28, 28), num_classes=10, enable_logging=False):
        """
        Builds a PyTorch model for image classification from the given config.
        Hyperparameters that aren't specified are taken from the self.default_config.

        :param config: dict-like object holding the architectural hyperparameters.
        :param input_shape: shape of single image
        :param num_classes: number of classes for classification
        :param enable_logging: indicates whether to create logs which get returned by self.retrieve_logs()
        """
        super(GrowableModel, self).__init__()

        # limiting the potential for shooting oneself in the foot
        assert 'n_conv_layers' in config
        assert config['n_conv_layers'] >= 1 and "no convolution not considered"
        assert sum(['trigger_threshold' in config, 'conv_trigger_threshold' in config]) == 1
        dropout_rate = config['dropout_rate'] if 'dropout_rate' in config else 0.0
        assert dropout_rate == 0.0 and "dropout was not tested yet"

        # fill unspecified config values with defaults
        full_config = self.default_config.copy()
        full_config.update(config)

        # read from config
        self.reproduce_paper = bool(full_config['reproduce_paper'])
        self.max_params = full_config['max_params']
        n_conv_layers = full_config['n_conv_layers']
        n_fc_layers = full_config['n_fc_layers']
        kernel_size = full_config['kernel_size']
        glob_av_pool = full_config['global_avg_pooling']
        max_pool = full_config['max_pool']
        use_BN = full_config['use_BN']
        in_channels_initial = in_channels_max = input_shape[0]
        key_conv = 'n_channels_conv_'
        key_fc = 'n_features_fc_'
        # edim_threshold = 1. - full_config['trigger_threshold_gap']
        if 'conv_trigger_threshold' in config:
            conv_trigger_threshold = full_config['conv_trigger_threshold']
            if n_fc_layers > 0:
                fc_trigger_threshold = full_config['fc_trigger_threshold']
        else:
            conv_trigger_threshold = fc_trigger_threshold = full_config['trigger_threshold']
        global out_channels_max_list
        out_channels_max_list = []
        # create convolutional layers (which go in the front)
        conv_list = []
        for i in range(n_conv_layers):
            out_channels_initial, out_channels_max = full_config[key_conv + str(i)]
            out_channels_max_list.append(out_channels_max)
            padding = (kernel_size - 1) // 2
            conv_0 = GrowableConv2d(in_channels_max, out_channels_max, in_channels_initial, out_channels_initial,
                                    kernel_size=kernel_size, padding=padding)
            block_layers = [conv_0, nn.ReLU(inplace=False)]
            # if i == 0:
            #     block_layers = [GrowableBatchNorm2d(in_channels_max, in_channels_initial, track_running_stats=False)] + block_layers
            if use_BN:
                block_layers.append(GrowableBatchNorm2d(out_channels_max, out_channels_initial, track_running_stats=False))
            if i not in [2,4,6]: #Comment for original arch
                block_layers.append(nn.MaxPool2d(kernel_size=2, stride=2,padding=padding))
            conv_block_0 = GrowableBlock(*block_layers,
                                         trigger_threshold=conv_trigger_threshold, enable_logging=enable_logging,
                                         reproduce_paper=self.reproduce_paper)
            conv_list.append(conv_block_0)

            in_channels_initial, in_channels_max = out_channels_initial, out_channels_max

        self.conv_layers = nn.ModuleList(conv_list)
        self.conv_activations = None  # a list holding the most recent activations

        # transition from convolutional layers to fully-connected (fc) layers
        # self.avgpooling = nn.AdaptiveAvgPool2d(1) if glob_av_pool else nn.Identity()
        self.pooling = nn.MaxPool2d(kernel_size,2,padding = (kernel_size - 1) // 2) if max_pool else nn.Identity() #changed

        in_features_initial = int(self._get_conv_output(input_shape))
        in_features_max = int(in_features_initial * out_channels_max / out_channels_initial)
        features_per_channel = in_features_initial // out_channels_initial
        assert out_channels_initial * features_per_channel == in_features_initial

        # create fully-connected layers (which come after the conv. ones)
        fc_list = []
        for i in range(n_fc_layers):
            out_features_initial, out_features_max = full_config[key_fc + str(i)]

            fc_0 = GrowableLinear(in_features_max, out_features_max, in_features_initial, out_features_initial,
                                  growth_multiplier=features_per_channel if i == 0 else 1)
            block_layers = [fc_0, nn.ReLU(inplace=False)]
            if use_BN:
                block_layers.append(GrowableBatchNorm1d(out_features_max, out_features_initial, track_running_stats=False))
            fc_block_0 = GrowableBlock(*block_layers,
                                       trigger_threshold=fc_trigger_threshold, enable_logging=enable_logging,
                                       reproduce_paper=self.reproduce_paper)
            fc_list.append(fc_block_0)

            in_features_initial, in_features_max = out_features_initial, out_features_max

        self.fc_layers = nn.ModuleList(fc_list)
        self.fc_activations = None  # a list holding the most recent activations

        # package last fc layer as block, so that its input may grow
        self.output_size = num_classes
        last_fc_0 = GrowableLinear(in_features_max, self.output_size, in_features_initial, self.output_size,
                                      growth_multiplier=features_per_channel if n_fc_layers==0 else 1)
        self.last_fc = GrowableBlock(last_fc_0, trigger_threshold=1., enable_logging=False,
                                     reproduce_paper=self.reproduce_paper)

        self.grow = False  # grow == True will cause activations to be saved during forward pass
        self.time_train = 0
        self.logging = enable_logging
        self.train_acc_log = []
        self.train_loss_log = []

    def _get_conv_output(self, shape):
        """Generate input sample and forward to get shape."""
        bs = 1
        x = Variable(torch.rand(bs, *shape))
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            # x = self.pooling(x)
        # x = self.pooling(x) #Uncomment for original arch
        n_size = x.data.view(bs, -1).size(1)
        return n_size

    def num_growing_params(self):
        """Returns the total number of parameters (or 'weights') in all growable layers."""
        num_params = sum([l.num_params() for l in self.conv_layers])
        num_params += sum([l.num_params() for l in self.fc_layers])
        num_params += self.last_fc.num_params()
        return num_params

    def max_params_reached(self):
        return self.max_params is not None and self.num_growing_params() >= self.max_params

    def room_to_max_params(self):
        return self.max_params - self.num_growing_params()

    def forward(self, x):
        # pass through convolutional layers
        if self.grow:
            self.conv_activations = [x.detach().clone()]
            # print('conv_activations',self.conv_activations)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            if self.grow:
                self.conv_activations.append(x.detach().clone())
            # x = self.maxpool(x)
            # x = self.pooling(x)

        # transition from 3d activations to 1d
        # x = self.pooling(x) #Uncomment for original arch
        # x = torch.flatten(x)
        x = x.view(x.size(0), -1)

        # pass through dense layers
        if self.grow:
            self.fc_activations = [x.detach().clone()]
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            if self.grow:
                self.fc_activations.append(x.detach().clone())

        # final layer
        x = self.last_fc(x)
        return x
    def grow_all_layers(self,edim_track,to_add_track):
        only_logging = self.max_params_reached() or (len(self.train_loss_log) % 10 != 0 and not self.reproduce_paper)
        # global edim_track
        # edim_track = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        # edim_track = None 
        # to_add_track = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        # to_add_track = None 
        # grow last layers first, as growing affects the input of the next layer
        def grow_layers(layers, activations, next_layer,edim_track,to_add_track):
            def get_next_layer(i):
                if i == len(layers) - 1:
                    return next_layer
                else:
                    return layers[i + 1]

            for i in reversed(range(len(layers))):
                if self.max_params is not None:
                    # neuron*2 in layer i -> params*2 in layer i and i+1
                    neurons_per_param = layers[i].num_neurons() / (layers[i].num_params() + get_next_layer(i).num_params())
                    max_added_neurons = math.floor(self.room_to_max_params() * neurons_per_param)
                else:
                    max_added_neurons = None
                # print(layers[i].num_params())

                added_neurons,edim = layers[i].grow(activations[i], activations[i + 1],
                                               max_growth=max_added_neurons, only_logging=only_logging)
                # print(added_neurons)
                # if i==7:
                # print(edim_track)
                edim_track[i].append(edim.item())
                to_add_track[i].append(added_neurons)
                # to_add_track[i].append(added_neurons)
                # print('edim',edim_track)
                # print('to_add',to_add_track)
                # extend the follow layer to accept the larger input (weights initialized to 0)
                if added_neurons:
                    get_next_layer(i).grow_input_by(added_neurons)
                # print(layers[i].num_params())
            return edim_track,to_add_track

        # x = grow_layers(self.fc_layers, self.fc_activations, self.last_fc)
        edim_track,to_add_track = grow_layers(self.conv_layers, self.conv_activations,
                    self.fc_layers[0] if self.fc_layers else self.last_fc,edim_track,to_add_track)
        return edim_track,to_add_track

    def train_fn(self, optimizer, criterion, loader, device, epoch,edim_track,to_add_track,track_growth,standard_channel_list,train=True, grow=True):
        """
        Training method
        :param optimizer: optimization algorithm
        :param criterion: loss function
        :param loader: data loader for either training or testing set
        :param device: torch device
        :param train: boolean to indicate if training or test set is used
        :param grow: boolean to indicate whether the model should grow during training
        :return: (accuracy, loss) on the data
        """
        time_begin = time.time()
        score = AvgrageMeter()
        objs = AvgrageMeter()

        self.train()
        self.grow = grow  # and not self.max_params_reached()
        # print(self.conv_layers[0].num_neurons())
        t = tqdm(loader)
        # edim_track = []
        # to_add_track = []
        # edim_track = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        # to_add_track = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        # print(len(t))
        for i,(images, labels) in enumerate(t):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = self(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if i%100==99:
                writer.add_scalar("training/learning_rate",optimizer.param_groups[0]['lr'],i+len(loader)*(epoch))
                writer.add_scalar("training/loss",loss/100,i+len(loader)*(epoch))
            # if i%50==0:

            if self.grow:
                edim_track,to_add_track = self.grow_all_layers(edim_track,to_add_track)
                # print(edim)
                # print(to_add)
                # print(layer)
                # edim_track[layer].append(edim.item())
                # to_add_track[layer].append(to_add)
                # print(edim)
            if self.max_params_reached():
                self.grow = False
            # to_add_track.append(to_add)
            # edim_track.append(edim)
            acc = accuracy(logits, labels, topk=(1,))[0]  # accuracy given by top 3
            n = images.size(0)
            objs.update(loss.item(), n)
            score.update(acc.item(), n)


            conv_channel_str = '-'.join([str(conv_layer.num_neurons()) for conv_layer in self.conv_layers])
            for i in range(len(self.conv_layers)):
                track_growth[i].append(self.conv_layers[i].num_neurons()/standard_channel_list[i])
            # print(self.conv_layers)
            # print(self.conv_layers[0].num_neurons().item())
            fc_channel_str = '-'.join([str(fc_layer.num_neurons()) for fc_layer in self.fc_layers])
            channel_str = f'conv[{conv_channel_str}]-fc[{fc_channel_str}]'
            t.set_description(f'(=> Training {channel_str} {self.num_growing_params():.2e}) Loss: {objs.avg:.4f}')

            if self.logging:
                self.train_acc_log.append(acc.item())
                self.train_loss_log.append(loss.item())

        self.time_train += time.time() - time_begin
        print('training time: ' + str(self.time_train))
        # print(self.out_channels_max)
        conv_channel_list = [conv_layer.num_neurons() for conv_layer in self.conv_layers]
        fc_channel_list = [fc_layer.num_neurons() for fc_layer in self.fc_layers]
        # full_config = self.default_config.copy()
        # full_config.update(config)
        # key_conv = 'n_channels_conv_'
        # out_max_channels_list = [] 
        # for i in range(8):
        #     out_channels_initial, out_channels_max = full_config[key_conv + str(i)]
        #     out_max_channels_list.append(out_channels_max)
        return score.avg, objs.avg, conv_channel_list, out_channels_max_list, fc_channel_list,edim_track,to_add_track,track_growth

    def eval_fn(self, loader, device, criterion,train=False):
        """
        Evaluation method
        :param loader: data loader for either training or testing set
        :param device: torch device
        :param train: boolean to indicate if training or test set is used
        :return: accuracy on the data
        """
        score = AvgrageMeter()
        self.eval()
        self.grow = False

        t = tqdm(loader)
        with torch.no_grad():  # no gradient needed
            for images, labels in t:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)
                loss = criterion(outputs,labels)
                acc = accuracy(outputs, labels, topk=(1,))[0]
                score.update(acc.item(), images.size(0))

                t.set_description('(=> Test) Score: {:.4f}'.format(score.avg))
                writer.add_scalar("val/loss",loss/100,len(loader))

        return score.avg

    def retrieve_logs(self):
        """Returns a dictionary which contains logs from layers etc."""
        logs = {'train_acc': self.train_acc_log.copy(),
                'train_loss': self.train_loss_log.copy(),
                'num_params': int(self.num_growing_params())}
        for i, conv_layer in enumerate(self.conv_layers):
            for k, v in conv_layer.retrieve_logs().items():
                logs[f'conv{i}_{k}'] = v
        for i, fc_layer in enumerate(self.fc_layers):
            for k, v in fc_layer.retrieve_logs().items():
                logs[f'fc{i}_{k}'] = v
        return logs

