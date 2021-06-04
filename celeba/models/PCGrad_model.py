import os
import random
import numpy as np
import torch
from .utils import get_blocks, create_optimizer
from .cnn import BasicNet
from .pcgrad import PCGrad

class PCGrad_model(BasicNet):
    def __init__(self, 
                 task_groups, 
                 opt):
        super(PCGrad_model, self).__init__(task_groups)
        self.one_optim_per_task = opt.one_optim_per_task
        self.learning_rate = opt.learning_rate

        # Optimizer
        if self.one_optim_per_task:
            self.optimizer = []
            for k in range(len(task_groups)):
                self.optimizer.append(PCGrad(create_optimizer(opt.optimizer, self.parameters(), self.learning_rate)))
        else:
            self.optimizer = PCGrad(create_optimizer(opt.optimizer, self.parameters(), self.learning_rate))
        
    def optim_step(self, task=None):
        if self.one_optim_per_task:
            self.optimizer[task].step()
        else:
            self.optimizer.step()
            
    def optim_zero_grad(self, task=None):
        if self.one_optim_per_task:
            self.optimizer[task].zero_grad()
        else:
            self.optimizer.zero_grad()
            
    def train_step(self, 
                   data, 
                   gts, 
                   loss_func):
        # Forward pass
        logits = self.forward(data)
        preds = [(elt>=0.5).type(torch.float32).detach() for elt in logits]
        task_losses = loss_func(logits, gts)

        # Backward pass
        self.optimizer.pc_backward(task_losses)
        self.optim_step()

        # Incr iter nb
        self.n_iter += 1
        task_losses = task_losses.detach()
        
        return task_losses, preds

    

    def test_step(self, 
                  data, 
                  gts, 
                  loss_func):
        # Forward pass
        logits = self.forward(data)
        preds = [(elt>=0.5).type(torch.float32) for elt in logits]
        task_losses = loss_func(logits, gts)
        loss = task_losses.sum()
        
        return task_losses, preds
    
    def initialize(self, 
                   opt, 
                   device, 
                   model_dir, 
                   saver):
        super(PCGrad_model, self).initialize(opt, 
                                          device, 
                                          model_dir)
        # Recovers an identical model and checkpoint
        if opt.recover:
            source_dir = os.path.join(opt.checkpoint_path, opt.reco_name) if opt.reco_name else model_dir
            ckpt_file = os.path.join(source_dir, opt.reco_type+'_weights.pth')
            ckpt = torch.load(ckpt_file, map_location=device)
            # Recovers optimizers if existing
            if 'optimizer_state_dict' in ckpt:
                if self.one_optim_per_task:
                    for k in range(len(ckpt['optimizer_state_dict'])):
                        self.optimizer._optim[k].load_state_dict(ckpt['optimizer_state_dict'][str(k)])
                        print('Optimizer {} recovered from {}.'.format(k, ckpt_file))
                else:
                    self.optimizer._optim.load_state_dict(ckpt['optimizer_state_dict'])
                    print('Optimizer recovered from {}.'.format(ckpt_file))
            # Recovers saver
            saver.load()
