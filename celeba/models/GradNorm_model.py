import os
import random
import numpy as np
import torch
from .utils import get_blocks, create_optimizer
from .cnn import BasicNet

    
class GradNorm_model(BasicNet):
    def __init__(self, 
                 task_groups, 
                 opt):
        super(GradNorm_model, self).__init__(task_groups)
        self.one_optim_per_task = opt.one_optim_per_task
        self.learning_rate = opt.learning_rate
        self.task_weights = torch.nn.Parameter(torch.ones(self.n_tasks, dtype=torch.float32), requires_grad=True)
        self.alpha = opt.alpha
        self.norms = torch.zeros(self.n_tasks)
        self.task_losses = torch.zeros(self.n_tasks)
        self.initial_losses = torch.zeros(self.n_tasks)
        
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
        # Forward pass
        logits = self.forward(data)
        preds = [(elt>=0.5).type(torch.float32).detach() for elt in logits]
        self.task_losses = loss_func(logits, gts)
        loss = torch.sum(self.task_weights*self.task_losses)

        # Backward pass
        self.optim_zero_grad()
        loss.backward(retain_graph=True)

        # zero the w_i(t) gradients 
        self.task_weights.grad = 0.0 * self.task_weights.grad
        W = self.get_weights()[-1]
        self.norms = []

        for w_i, L_i in zip(self.task_weights, self.task_losses):
            # gradient of L_i(t) w.r.t. W
            gLgW = torch.autograd.grad(L_i, W, retain_graph=True)

            # G^{(i)}_W(t)
            self.norms.append(torch.norm(w_i * gLgW[0]))

        self.norms = torch.stack(self.norms)

        # set L(0)
        # if using log(C) init, remove these two lines
        if self.n_iter == 0:
            self.initial_losses = self.task_losses.detach()

        # compute the constant term without accumulating gradients
        # as it should stay constant during back-propagation
        with torch.no_grad():

            # loss ratios \curl{L}(t)
            loss_ratios = self.task_losses / self.initial_losses

            # inverse training rate r(t)
            inverse_train_rates = loss_ratios / loss_ratios.mean()
            constant_term = self.norms.mean() * (inverse_train_rates ** self.alpha)

        # write out the gradnorm loss L_grad and set the weight gradients
        grad_norm_loss = (self.norms - constant_term).abs().sum()
        self.task_weights.grad = torch.autograd.grad(grad_norm_loss, self.task_weights)[0]

        # apply gradient descent
        self.optim_step()

        # renormalize the gradient weights
        with torch.no_grad():

            normalize_coeff = len(self.task_weights) / self.task_weights.sum()
            self.task_weights.data = self.task_weights.data * normalize_coeff

        # Detach losses
        self.task_losses = self.task_losses.detach()

        # Incr iter nb
        self.n_iter += 1
        
        return self.task_losses, preds

    

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
    
    def checkpoint(self):
        ckpt = super(GradNorm_model, self).checkpoint()
        ckpt['initial_losses'] = self.initial_losses
        ckpt['task_losses'] = self.task_losses
        ckpt['norms'] = self.norms
        return ckpt

    def initialize(self,
                   opt,
                   device,
                   model_dir,
                   saver):
        super(GradNorm_model, self).initialize(opt, 
                                              device, 
                                              model_dir)
        # Recovers an identical model and checkpoint
        if opt.recover:
            source_dir = os.path.join(opt.checkpoint_path, opt.reco_name) if opt.reco_name else model_dir
            ckpt_file = os.path.join(source_dir, opt.reco_type+'_weights.pth')
            ckpt = torch.load(ckpt_file, map_location=device)
            pretrained_dict = {k:v for k,v in ckpt['model_state_dict'].items() if 'task_weights' in k}
            if len(pretrained_dict) > 0:
                self.load_state_dict(pretrained_dict, strict=False)
                print('Tasks weights recovered from {}.'.format(ckpt_file))
            else:
                print('No tasks weights found in {}.'.format(ckpt_file))
            if 'initial_losses' in ckpt:
                self.initial_losses = ckpt['initial_losses']
                print('Initial losses recovered from {}.'.format(ckpt_file))
            else:
                print('No initial losses found in {}.'.format(ckpt_file))
            if 'task_losses' in ckpt:
                self.task_losses = ckpt['task_losses']
                print('Task losses recovered from {}.'.format(ckpt_file))
            else:
                print('No task losses found in {}.'.format(ckpt_file))
            if 'norms' in ckpt:
                self.norms = ckpt['norms']
                print('Norms recovered from {}.'.format(ckpt_file))
            else:
                print('No norms found in {}.'.format(ckpt_file))
                
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
                
