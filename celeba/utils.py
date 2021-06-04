import os
import math
import numpy as np
import subprocess as sp
from models import *

def check_gpu():
    """
    Selects an available GPU
    """
    available_gpu = -1
    ACCEPTABLE_USED_MEMORY = 500
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    memory_used_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    for k in range(len(memory_used_values)):
        if memory_used_values[k]<ACCEPTABLE_USED_MEMORY:
            available_gpu = k
            break
    return available_gpu


def create_task_groups(groups_size):
    """
    Creates random task groups of given size.
    """
    if groups_size==1:
        task_groups = [[k] for k in range(40)]
    else:
        tasks = [k for k in range(40)]
        nb_groups = math.ceil(40./float(groups_size))
        task_groups = []
        for group_id in range(nb_groups):
            group = np.random.choice(tasks, min(groups_size, len(tasks)), replace=False).tolist()
            for elt in group:
                tasks.remove(elt)
            task_groups.append(group)
    return task_groups

def select_model(opt, task_groups):
    """
    Select the model to use.
    """
    method = opt.method.upper()
    if method == 'MTL':
        model = MTL_model(task_groups, opt)
    if method == 'MGDA':
        model = MGDA_model(task_groups, opt)
    if method == 'PCGRAD':
        model = PCGrad_model(task_groups, opt)
    if method == 'GRADNORM':
        model = GradNorm_model(task_groups, opt)
    if method == 'MR':
        model = MR_model(task_groups, opt)
    return model
        
        
class logger():
    """
    Simple logger, which display the wanted metrics in a proper format.
    """
    def __init__(self, metrics, to_disp=['fscore', 'avg_error']):
        self.metrics=metrics
        self.to_disp=to_disp
        
    def log(self, n_epoch, n_iter, train_metrics, val_metrics, train_loss, val_loss):
        print('EVAL epoch {} iter {}: '.format(n_epoch, n_iter), ' '*50)
        for k in range(len(self.metrics)):
            if self.metrics[k] in self.to_disp:
                print('{} : {:.4f} / {:.4f}'.format(self.metrics[k], train_metrics[k], val_metrics[k]))
        print('_'*50)
                  
              
    