import os
import random
import numpy as np
import torch
from .utils import get_blocks, create_optimizer
from .cnn import BasicNet


class MTL_model(BasicNet):
    def __init__(self, 
                 task_groups, 
                 opt):
        super(MTL_model, self).__init__(task_groups)
        self.one_optim_per_task = opt.one_optim_per_task
        self.learning_rate = opt.learning_rate
        self.trace_param = opt.trace_param

        # Optimizer
        if self.one_optim_per_task:
            self.optimizer = []
            for k in range(len(task_groups)):
                self.optimizer.append(create_optimizer(opt.optimizer, self.parameters(), self.learning_rate))
        else:
            self.optimizer = create_optimizer(opt.optimizer, self.parameters(), self.learning_rate)
            
        # Params trace
        if self.trace_param:
            self.param_init = [v.data.numpy() for (k,v) in self.named_parameters() if (not 'out' in k and not 'BN' in k)]
            self.current_param = [v.data for (k,v) in self.named_parameters() if (not 'out' in k and not 'BN' in k)]
            self.abs_distance_L2 = torch.zeros((), dtype=torch.float64) 
            np.save(os.path.join(opt.checkpoint_path, opt.name, 'init_weights.npy'), self.param_init)
        
        
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
        # Tasks optimized separately
        task_losses = [None]*self.n_tasks
        preds = [None]*self.n_tasks

        # Inference
        for task in range(self.n_tasks):
            # Forward pass
            logits = self.forward(data, task=task)
            preds[task] = (logits>=0.5).type(torch.float32).detach()
            task_loss = loss_func(logits, gts, task)
            
            # Backward pass
            self.optim_zero_grad(task)
            task_loss.backward()
            self.optim_step(task)
            task_losses[task] = task_loss.detach()
            
            # Params trace
            if self.trace_param:
                new_param = [v.data.cpu() for (k,v) in self.named_parameters() if (not 'out' in k and not 'BN' in k)]
                abs_dist = [((new_param[k] - self.current_param[k])**2).sum() for k in range(len(new_param))]
                abs_dist = torch.sqrt(sum(abs_dist))
                self.abs_distance_L2 += abs_dist
                self.current_param = new_param

        # Incr iter nb
        self.n_iter += 1
        task_losses = torch.stack(task_losses)

            
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
        super(MTL_model, self).initialize(opt, 
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
                        self.optimizer[k].load_state_dict(ckpt['optimizer_state_dict'][str(k)])
                        print('Optimizer {} recovered from {}.'.format(k, ckpt_file))
                else:
                    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    print('Optimizer recovered from {}.'.format(ckpt_file))
            # Recovers saver
            saver.load()

            
