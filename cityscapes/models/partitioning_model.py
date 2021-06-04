import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from .utils import get_blocks, create_optimizer
from .unet import UNet
from .partitionings import MRPartitioning



class PartitioningModel(UNet):
    def __init__(self, 
                 task_groups, 
                 partitioning, 
                 opt,
                 bilinear=True):
        super(PartitioningModel, self).__init__(task_groups, opt, partitioning, bilinear)
        self.per_batch_step = opt.per_batch_step
        self.one_optim_per_task = opt.one_optim_per_task
        self.learning_rate = opt.learning_rate

        # Optimizer
        if self.one_optim_per_task and not self.per_batch_step:
            self.optimizer = []
            for k in range(len(task_groups)):
                self.optimizer.append(create_optimizer(opt.optimizer, self.parameters(), self.learning_rate))
        else:
            self.optimizer = create_optimizer(opt.optimizer, self.parameters(), self.learning_rate)
    
    def optim_step(self, task=None):
        if self.one_optim_per_task and not self.per_batch_step:
            self.optimizer[task].step()
        else:
            self.optimizer.step()
            
    def optim_zero_grad(self, task=None):
        if self.one_optim_per_task and not self.per_batch_step:
            self.optimizer[task].zero_grad()
        else:
            self.optimizer.zero_grad()
            
    def update(self):
        pass
    
    def train_step(self, 
                   data, 
                   gts, 
                   loss_func):
        # Tasks optimized altogether
        if self.per_batch_step:
            task_losses = [None]*self.n_tasks
            preds = [None]*self.n_tasks

            # Tasks order
            task_ids = range(self.n_tasks)

            # Inference
            self.optim_zero_grad()
            for task in task_ids:
                self.change_task(task)

                # Forward pass
                logits = self.forward(data, task=task)
                task_loss = loss_func(logits, gts, group=task)

                # Acc. Backward pass
                task_loss.backward()
                task_losses[task] = task_loss.detach()
                preds[task] = logits.detach()

            # Update
            self.optim_step()
            
            # Incr iter nb
            self.n_iter += 1
            task_losses = torch.stack(task_losses)

        # Tasks optimized separately
        else:
            task_losses = [None]*self.n_tasks
            preds = [None]*self.n_tasks

            # Tasks order
            task_ids = range(self.n_tasks)

            # Inference
            for task in task_ids:
                self.change_task(task)

                # Forward pass
                logits = self.forward(data, task=task)
                task_loss = loss_func(logits, gts, group=task)

                # Backward pass
                self.optim_zero_grad(task)
                task_loss.backward()
                self.optim_step(task)
                task_losses[task] = task_loss.detach()
                preds[task] = logits.detach()

            # Incr iter nb
            self.n_iter += 1
            task_losses = torch.stack(task_losses)

        return task_losses, preds

    def test_step(self, data, gts, loss_func):
        task_losses = [None]*self.n_tasks
        preds = [None]*self.n_tasks
        
        for task in range(self.n_tasks):
            self.change_task(task)

            # Forward pass
            logits = self.forward(data, task)
            task_loss = loss_func(logits, gts, group=task)
            task_losses[task] = task_loss
            preds[task] = logits.detach()
        
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
        
