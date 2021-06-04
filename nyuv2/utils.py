import os
import subprocess as sp
from models import *

CLASS_NB = 13

def check_gpu():
    """
    Selects an available GPU
    """
    available_gpu = -1
    ACCEPTABLE_USED_MEMORY = 1000
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    memory_used_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    for k in range(len(memory_used_values)):
        if memory_used_values[k]<ACCEPTABLE_USED_MEMORY:
            available_gpu = k
            break
    return available_gpu


    
def create_task_groups(opt):
    """
    Creates task groups depending on user inputs.
    """
    nll = opt.seg_loss.upper()=='NLL'
    task_groups = []
    n_groups = 0
    # Argmax segmentation predictions. We add a background class.
    if nll:
        task_groups = [{'ids': list(range(-1, CLASS_NB)),
                        'type': 'segmentation',
                        'size': CLASS_NB+1}]
    # Binary class segmentation predictions. No need for background.
    else:
        # Treats all segmentation tasks independently.
        if opt.idp_seg_tasks:
            for k in range(CLASS_NB):
                task_groups.append({'ids': [k],
                                    'type': 'segmentation',
                                    'size': 1})
        # Else, learns them altogether.
        else:
            task_groups = [{'ids': list(range(CLASS_NB)),
                            'type': 'segmentation',
                            'size': CLASS_NB}]
    n_groups +=1
    # Depth and normals tasks.
    task_groups.append({'ids':[CLASS_NB], 
                        'type': 'depth', 
                        'size': 1})
    n_groups +=1
    task_groups.append({'ids':[CLASS_NB+1], 
                        'type': 'normals', 
                        'size': 3})
    n_groups +=1
    return task_groups, n_groups, len(task_groups)


def select_model(opt, task_groups):
    """
    Select the model to use.
    """
    method = opt.method.upper()
    if method == 'MTL':
        model = MTL_model(task_groups, opt)
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
    def __init__(self, metrics, to_disp=['mIoU', 'Excl_mIoU', 'Pix_Acc', 'Abs_Error', 'Mean_Error']):
        self.metrics=metrics
        self.to_disp=to_disp
        
    def log(self, n_epoch, n_iter, train_metrics, val_metrics, train_loss, val_loss):
        print('EVAL epoch {} iter {}: '.format(n_epoch, n_iter), ' '*50)
        for k in range(len(self.metrics)):
            if self.metrics[k] in self.to_disp:
                print('{} : {:.4f} / {:.4f}'.format(self.metrics[k], train_metrics[k], val_metrics[k]))
        print('_'*50)
                  
              
    