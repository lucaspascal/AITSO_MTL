import os
import random
import numpy as np
import torch
from .utils import get_blocks, create_optimizer
from .cnn import BasicNet
from torch.autograd import Variable
from .min_norm_solvers import MinNormSolver, gradient_normalizers

class MGDA_model(BasicNet):
    def __init__(self, 
                 task_groups,
                 opt):
        super(MGDA_model, self).__init__(task_groups)
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
            
    def train_step(self, 
                   data, 
                   gts, 
                   loss_func):
        loss_data = {}
        scale = {}
        grads = {}
        task_losses = [None]*self.n_tasks
        preds = [None]*self.n_tasks
        
        # Forward shared pass without gradients
        self.optim_zero_grad()
        with torch.no_grad():
            feats = self.forward_shared(data)
        rep_variable = Variable(feats.data.clone(), requires_grad=True)
        
        # Tasks forward passes with backprop on shared weights
        for t in range(self.n_tasks):
            self.optim_zero_grad()
            out_t = self.forward_task(rep_variable, t)
            preds[t] = (out_t>=0.5).type(torch.float32).detach()
            task_loss = loss_func(out_t, gts, active_task=t)
            task_losses[t] = task_loss.detach()
            loss_data[t] = task_loss.data
            task_loss.backward()
            grads[t] = [] 
            grads[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
            rep_variable.grad.data.zero_()
            
            
        # Normalize all gradients, this is optional and not included in the paper.
        gn = gradient_normalizers(grads, loss_data, 'none')
        for t in range(self.n_tasks):
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        # Frank-Wolfe iteration to compute scales.
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in range(self.n_tasks)])
        for t in range(self.n_tasks):
            scale[t] = float(sol[t])            
            
            
            
        # Scaled back-propagation
        self.optim_zero_grad()
        feats = self.forward_shared(data)
        for t in range(self.n_tasks):
            out_t = self.forward_task(feats, t)
            loss_t = loss_func(out_t, gts, active_task=t)
            loss_data[t] = loss_t.data
            if t > 0:
                loss = loss + scale[t]*loss_t
            else:
                loss = scale[t]*loss_t
        loss.backward()
        self.optim_step()

            
        # Incr iter nb
        self.n_iter += 1
        task_losses = torch.stack(task_losses)
                
        return task_losses, preds

    

    def test_step(self, data, gts, loss_func):
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
        super(MGDA_model, self).initialize(opt, 
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

            
