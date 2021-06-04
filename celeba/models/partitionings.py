import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class Partitioning(nn.Module):
    """ 
    Applies task specific masking out individual units in a layer.
    Args:
    unit_count  (int): Number of input channels going into the layer.
    task_count  (int): Number of tasks.
    sigma (int): Sharing ratio.
    """

    def __init__(self, active_task=0):
        super(Partitioning, self).__init__()
        self.active_task=active_task

    def set_active_task(self, active_task):
        """
        Changes the current active task.
        """
        self.active_task = active_task
        return active_task
    
    def forward(self, x, hard=False):
        """
        Forward pass.
        """
        pass          
            

        
class MRPartitioning(Partitioning):
    """ 
    Applies task specific masking out individual units in a layer.
    Args:
    unit_count  (int): Number of input channels going into the layer.
    task_count  (int): Number of tasks.
    sigma (int): Sharing ratio.
    """

    def __init__(self, unit_count, task_count, sigma, active_task=0):
        super(MRPartitioning, self).__init__(active_task)
        self.task_count = task_count
        self.unit_count = unit_count
        # This implementation deals with 1-sigma instead of sigma.
        self.sigma = 1.-sigma

        # Catches sigma=0 case 
        if self.sigma == 0:
            self.unit_mapping = np.ones((task_count, unit_count), dtype=np.float32)

        else:
            # Initializes a minimal mask covering all filters with diagonal matrices
            self.unit_mapping = np.eye(task_count, self.unit_count, dtype=np.float32)
            idx = task_count
            while False in (np.sum(self.unit_mapping, axis=0)>0).tolist():
                self.unit_mapping += np.eye(task_count, self.unit_count, idx, dtype=np.float32)
                idx += task_count

            # Fills each line with zeros until the wanted ratio is reached
            for k in range(task_count):
                rep = np.sum(self.unit_mapping[k,:]==0) / self.unit_count
                while rep > self.sigma:
                    free_idx =  np.where(self.unit_mapping[k,:]==0)[0]
                    self.unit_mapping[k, free_idx[np.random.randint(0,len(free_idx))]] = 1
                    rep = np.sum(self.unit_mapping[k,:]==0) / self.unit_count

        # Store tested filter/task couples
        self.tested_tasks = torch.nn.Parameter(torch.from_numpy(self.unit_mapping>0), requires_grad=False)

        # Convert routing map to torch parameter
        self.unit_mapping = torch.nn.Parameter(torch.from_numpy(self.unit_mapping), requires_grad=False)


    def assign_mapping(self, new_mapping, reset=False):
        """
        Assign a new mapping to the MR layer.
        """
        device = self.unit_mapping.device
        self.unit_mapping = torch.nn.Parameter(torch.from_numpy(new_mapping).to(device), requires_grad=False)
        if reset:
            self.tested_tasks = torch.nn.Parameter(torch.from_numpy(new_mapping>0).to(device), requires_grad=False)
        else:
            self.tested_tasks = torch.nn.Parameter((self.tested_tasks.type(torch.bool) + (self.unit_mapping>0)).type(torch.bool), requires_grad=False)
            
    def get_unit_mapping(self):
        """
        Return the unit mapping of the layer.
        """
        return self.unit_mapping.cpu().numpy(), self.tested_tasks.cpu().numpy()
        
    def convert_to_diff(self, temp=5., temp_decay=0.965):
        new_unit_mapping = np.ones((self.task_count, self.unit_count,2), dtype=np.float32)*0.5
        self.unit_mapping = torch.nn.Parameter(torch.from_numpy(new_unit_mapping), requires_grad=True)#.to(device)
        self.temp = temp
        self.temp_decay = temp_decay
        

    def forward(self, x, hard=False):
        """
        Forward pass. Feature maps are selected (or zeroed) w.r.t. the current active task.
        """
        mask = self.unit_mapping[self.active_task,:].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = x*mask
            
        return x            
