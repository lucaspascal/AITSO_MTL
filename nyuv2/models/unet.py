import os
from collections import OrderedDict
from .utils import get_blocks
from .unet_parts import *

        
class UNet(nn.Module):
    def __init__(self, 
                 task_groups,
                 opt,
                 partitioning=None,
                 bilinear=True):
        super(UNet, self).__init__()
        self.task_groups = task_groups
        self.sigma = opt.sigma
        self.n_tasks = len(task_groups)
        self.bilinear = bilinear
        self.n_iter = 0
        self.n_epoch = 0
        self.size = 18

        # Convolutional body
        self.inc = DoubleConv(3, 
                              64, 
                              n_tasks=self.n_tasks,
                              partitioning=partitioning,
                              sigma=self.sigma)
        self.down1 = Down(64, 
                          128, 
                          n_tasks=self.n_tasks, 
                          partitioning=partitioning,
                          sigma=self.sigma)
        self.down2 = Down(128, 
                          256, 
                          n_tasks=self.n_tasks, 
                          partitioning=partitioning,
                          sigma=self.sigma)
        self.down3 = Down(256, 
                          512, 
                          n_tasks=self.n_tasks, 
                          partitioning=partitioning,
                          sigma=self.sigma)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 
                          1024 // factor, 
                          n_tasks=self.n_tasks, 
                          partitioning=partitioning,
                          sigma=self.sigma)
        self.up1 = Up(1024, 
                      512 // factor, 
                      n_tasks=self.n_tasks, 
                      partitioning=partitioning,
                      sigma=self.sigma,
                      bilinear=bilinear)
        self.up2 = Up(512, 
                      256 // factor, 
                      n_tasks=self.n_tasks, 
                      partitioning=partitioning,
                      sigma=self.sigma,
                      bilinear=bilinear)
        self.up3 = Up(256, 
                      128 // factor, 
                      n_tasks=self.n_tasks, 
                      partitioning=partitioning,
                      sigma=self.sigma,
                      bilinear=bilinear)
        self.up4 = Up(128, 
                      64, 
                      n_tasks=self.n_tasks, 
                      partitioning=partitioning,
                      sigma=self.sigma,
                      bilinear=bilinear)
        
        # Prediction head
        self.out_layer = []
        for task in range(len(task_groups)):
            self.out_layer.append(('OUTCONV_{}'.format(task), OutConv(64, task_groups[task]['size'])))
        self.out_layer = nn.Sequential(OrderedDict(self.out_layer))
        self.seg_activation = 'sigmoid' if opt.seg_loss.upper()=='BCE' else 'log_softmax'

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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x
    

    def forward_task(self, x, task):
        """
        Only computes a task-specific prediction, from a shared representation.
        """
        x = self.out_layer[task](x)
        if  self.task_groups[task]['type'] == 'segmentation':
            if self.seg_activation == 'sigmoid':
                x = torch.sigmoid(x)
            elif self.seg_activation == 'log_softmax':
                x = F.log_softmax(x, dim=1)
        elif  self.task_groups[task]['type'] == 'depth':
            None
        elif  self.task_groups[task]['type'] == 'normals':
            x = (x / torch.norm(x, p=2, dim=1, keepdim=True) + 1.)/2.
        return x
   
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
        blocks = [elt for elt in get_blocks(self, nn.Conv2d) if elt.out_channels>13]
        return blocks

    def get_weights(self):
        a=[elt.weight for elt in self.get_convs()]
        return [elt.weight for elt in self.get_convs()]

    def get_weight(self, depth):
        return self.get_weights()[depth]

    def get_BNs(self):
        return get_blocks(self, nn.BatchNorm2d)

    def get_conv(self, depth):
        return self.get_convs()[depth]

    def get_BN(self, depth):
        return self.get_BNs()[depth]
