import os
import random
import numpy as np
import torch
from torch.autograd import Variable
from .utils import get_blocks, create_optimizer
from .cnn import BasicNet
from .partitionings import MRPartitioning

class PartitioningModel(BasicNet):
    def __init__(self, 
                 task_groups,
                 partitioning,
                 opt):
        super(PartitioningModel, self).__init__(task_groups, 
                                                partitioning=partitioning, 
                                                sigma=opt.sigma)
        self.one_optim_per_task = opt.one_optim_per_task
        self.learning_rate = opt.learning_rate

        # Optimizer
        if self.one_optim_per_task:
            self.optimizer = []
            for k in range(len(task_groups)):
                self.optimizer.append(create_optimizer(opt.optimizer, self.parameters(), self.learning_rate))
        else:
            self.optimizer = create_optimizer(opt.optimizer, self.parameters(), self.learning_rate)
        
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
            
    def update(self):
        pass
    
    def train_step(self, 
                   data, 
                   gts, 
                   loss_func):
        task_losses = [None]*self.n_tasks
        preds = [None]*self.n_tasks

        # Tasks order
        task_ids = range(self.n_tasks)

        # Inference
        self.optim_zero_grad()
        for task in task_ids:
            self.change_task(task)

            # Forward pass
            logits = self.forward(data, task)
            preds[task] = (logits>=0.5).type(torch.float32).detach()
            task_loss = loss_func(logits, gts, task)

            # Backward pass
            task_loss.backward()
            task_losses[task] = task_loss.detach()

        # Update
        self.optim_step()

        # Incr iter nb
        self.n_iter += 1
        task_losses = torch.stack(task_losses)

        return task_losses, preds

    def test_step(self, data, gts, loss_func):
        task_losses = []
        preds = []
        for task in range(self.n_tasks):
            self.change_task(task)

            # Forward pass
            logits = self.forward(data, task)
            preds.append((logits>=0.5).type(torch.float32).detach())
            task_loss = loss_func(logits, gts, task)
            task_losses.append(task_loss.detach())
        
        return torch.stack(task_losses), preds
    
                
    def freeze_partitions(self):
        pass

    def change_task(self, task):
        def aux(m):
            if hasattr(m, 'active_task'):
                m.set_active_task(task)
        self.apply(aux)

    def set_active_task(self, active_task):
        self.active_task = active_task
        

    def initialize(self,
                   opt,
                   device,
                   model_dir):
        super(PartitioningModel, self).initialize(opt, 
                                                  device, 
                                                  model_dir)
        
