import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .utils import get_blocks
from .partitionings import MRPartitioning

class ConvLayer(nn.Module):
    """
    Convolutional layer, including BN, activation and Max Roaming layer.
    """
    def __init__(self, 
                 in_planes, 
                 out_planes, 
                 kernel_size=3, 
                 stride=1, 
                 padding=0, 
                 norm_layer=None, 
                 act_layer=None,
                 n_tasks=None, 
                 partitioning=None, 
                 sigma=None):
        
        super(ConvLayer, self).__init__()
        modules = [('CONV', nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                   stride=stride, padding=padding, bias=False))]
        if norm_layer is not None:
            modules.append(('BN', norm_layer(num_features=out_planes, track_running_stats=False)))
        if act_layer is not None:
            modules.append(('ACT', act_layer(inplace=True)))
        if sigma and partitioning=='MR':
            modules.append(('MR', MRPartitioning(out_planes, n_tasks, sigma)))
        self.conv_block = nn.Sequential(OrderedDict(modules))
        
        

    def forward(self, x):
        return self.conv_block(x)
    
    def get_weight(self):
        return self.conv_block[0].weight
        
    def get_routing_block(self):
        return self.conv_block[-1]
    
    def get_routing_masks(self):
        mapping = self.conv_block[-1].unit_mapping.detach().cpu().numpy()
        tested = self.conv_block[-1].tested_tasks.detach().cpu().numpy()
        return mapping, tested 
        
        
class FCLayer(nn.Module):
    """
    FC layer, including BN and activation.
    """
    def __init__(self, 
                 in_planes, 
                 out_planes, 
                 norm_layer=None, 
                 act_layer=None,
                 n_tasks=None, 
                 partitioning=None, 
                 sigma=None):
        
        super(FCLayer, self).__init__()
        modules = [('FC', nn.Linear(in_planes, out_planes))]
        if norm_layer is not None:
            modules.append(('BN', norm_layer(num_features=out_planes, track_running_stats=False)))
        if act_layer is not None:
            modules.append(('ACT', act_layer(inplace=True)))
        self.fc_block = nn.Sequential(OrderedDict(modules))

    def forward(self, x):
        return self.fc_block(x)
    
    def get_weight(self):
        return self.fc_block[0].weight
        
    def get_routing_block(self):
        return self.fc_block[-1]
    
    def get_routing_masks(self):
        mapping = self.fc_block[-1].unit_mapping.detach().cpu().numpy()
        tested = self.fc_block[-1].tested_tasks.detach().cpu().numpy()
        return mapping, tested 
        
        
class BasicNet(nn.Module):
    def __init__(self,
                 task_groups,
                 norm_layer=None,
                 partitioning=None,
                 sigma=None,
                ):
        super(BasicNet, self).__init__()
        self.n_tasks = len(task_groups)
        norm_layer_2d = nn.BatchNorm2d
        norm_layer_1d = nn.BatchNorm1d
        act_layer = nn.ReLU
        self.n_iter = 0
        self.n_epoch = 0
        self.size = 7

        # Convolutional body
        self.backbone = [('CONV_1', ConvLayer(3, 64, 3, 1, 1, norm_layer_2d, act_layer, self.n_tasks, partitioning, sigma)),
                         ('POOL_1', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)), 
                         ('CONV_2', ConvLayer(64, 128, 3, 1, 1, norm_layer_2d, act_layer, self.n_tasks, partitioning, sigma)),
                         ('CONV_3', ConvLayer(128, 128, 3, 1, 1, norm_layer_2d, act_layer, self.n_tasks, partitioning, sigma)),
                         ('POOL_2', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)), 
                         ('CONV_4', ConvLayer(128, 256, 3, 1, 1, norm_layer_2d, act_layer, self.n_tasks, partitioning, sigma)),
                         ('CONV_5', ConvLayer(256, 256, 3, 1, 1, norm_layer_2d, act_layer, self.n_tasks, partitioning, sigma)),
                         ('POOL_3', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)), 
                         ('CONV_6', ConvLayer(256, 512, 3, 1, 1, norm_layer_2d, act_layer, self.n_tasks, partitioning, sigma)),
                         ('CONV_7', ConvLayer(512, 512, 3, 1, 1, norm_layer_2d, act_layer, self.n_tasks, partitioning, sigma)),
                         ('POOL_4', nn.AdaptiveAvgPool2d(1)),
                        ]
        self.backbone = nn.Sequential(OrderedDict(self.backbone))

        # FC layers
        self.fc = [('FC_INT_1', FCLayer(512, 512, norm_layer_1d, act_layer, self.n_tasks, partitioning, sigma)), 
                   ('FC_INT_2', FCLayer(512, 512, norm_layer_1d, act_layer, self.n_tasks, partitioning, sigma)), 
                  ]
        self.fc = nn.Sequential(OrderedDict(self.fc))
        
        # Prediction head
        self.out_layer = []
        for task in range(len(task_groups)):
            self.out_layer.append(('FC_{}'.format(task), nn.Linear(512, len(task_groups[task]))))
        self.out_layer = nn.Sequential(OrderedDict(self.out_layer))

    def forward(self, x, task=None):
        x = self.forward_shared(x)
        if task==None:
            x = [self.forward_task(x, task) for task in range(self.n_tasks)]
        else:
            x = self.forward_task(x, task)            
        return x
    
    def forward_shared(self, x):
        """
        Only computes the shared parts of the networks.
        """
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward_task(self, x, task):
        """
        Only computes a task-specific prediction, from a shared representation.
        """
        out = torch.sigmoid(self.out_layer[task](x))

        return out
        

    def initialize(self, 
                   opt, 
                   device, 
                   model_dir):

        # Nothing if no load required
        if opt.recover:
            source_dir = os.path.join(opt.checkpoint_path, opt.reco_name) if opt.reco_name else model_dir
            ckpt_file = os.path.join(source_dir, opt.reco_type+'_weights.pth')
            ckpt = torch.load(ckpt_file, map_location=device)
            
            # Gets what needed in the checkpoint
            pretrained_dict = {k:v for k,v in ckpt['model_state_dict'].items() if 'CONV' in k or 'BN' in k or 'FC' in k}
            
            # Loads the weights
            self.load_state_dict(pretrained_dict, strict=False)
            print('Weights recovered from {}.'.format(ckpt_file))
            
            # For recovering only
            if opt.recover:
                if source_dir == model_dir:
                    self.n_epoch = ckpt['epoch']
                    self.n_iter = ckpt['n_iter'] + 1
            
    def checkpoint(self):
        # Prepares checkpoint
        ckpt = {'model_state_dict': self.state_dict(),
                'epoch': self.n_epoch,
                'n_iter': self.n_iter}
        return ckpt
   
            
    def get_convs(self):
        blocks = [elt for elt in get_blocks(self, nn.Conv2d) if elt.out_channels>1]
        return blocks

    def get_weights(self):
        return [elt.weight for elt in self.get_convs()]

    def get_weight(self, depth):
        return self.get_weights()[depth]

    def get_BNs(self):
        return get_blocks(self, nn.BatchNorm2d)

    def get_conv(self, depth):
        return self.get_convs()[depth]

    def get_BN(self, depth):
        return self.get_BNs()[depth]
        