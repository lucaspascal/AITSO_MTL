import os
import random
import numpy as np
import torch
import math
from .utils import get_blocks
from .partitioning_model import PartitioningModel
from .partitionings import MRPartitioning


class MR_model(PartitioningModel):
    '''
    Modified MR strategy which adapts the update interval of the different layers depending
    on their size to have them finishing the update process altogether.
    '''
    def __init__(self, 
                 task_groups, 
                 opt):
        super(MR_model, self).__init__(task_groups, 
                                      'MR', 
                                      opt)
        self.n_updates = 0
        self.updates_end = opt.updates_end
        self.update_ratio = opt.update_ratio
        self.updates_size = opt.updates_size
        
        self.update_intervals = []
        for k in range(self.size):
            routing_mask,_ = self.get_routing_mask(k)
            nb_updates = math.ceil((1-routing_mask).sum(axis=1).max() / self.updates_size)
            self.update_intervals.append(int(self.updates_end/nb_updates) if nb_updates>0 else np.inf)
            
            
    def initialize(self,
                   opt,
                   device,
                   model_dir,
                   saver):
        super(MR_model, self).initialize(opt, device, model_dir)

        # Recovers an identical model and checkpoint
        if opt.recover:
            source_dir = os.path.join(opt.checkpoint_path, opt.reco_name) if opt.reco_name else model_dir
            ckpt_file = os.path.join(source_dir, opt.reco_type+'_weights.pth')
            ckpt = torch.load(ckpt_file, map_location=device)
            pretrained_dict = {k:v for k,v in ckpt['model_state_dict'].items() if 'MR' in k}
            # Recovers MR partitions
            if len(pretrained_dict) > 0:
                self.load_state_dict(pretrained_dict, strict=False)
                print('Partitions recovered from {}.'.format(ckpt_file))
            else:
                print('No partitions found in {}.'.format(ckpt_file))
               
            # Recovers optimizer, if existing
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
        
    def update(self):
        for depth in range(self.size):
            if self.n_iter % self.update_intervals[depth] == 0 and self.n_iter <= self.updates_end:
                routing_mask, tested_units = self.get_routing_mask(depth) # Get current mask and history mask
                # If update ratio reached, pass
                nb = routing_mask.shape[1]
                nb_used = tested_units.sum(axis=1)[0]
                nb_in_use = routing_mask.sum(axis=1)[0]
                ratio = (nb_used-nb_in_use)/(nb-nb_in_use)
                if ratio >= self.update_ratio:
                    continue
                
                # Updates the mapping
                new_mapping = self.updated_mapping(routing_mask, tested_units)
                MR_layer = self.get_MR(depth)
                MR_layer.assign_mapping(new_mapping)

                
    def updated_mapping(self, routing_mask, tested_units):
        update_size = min(np.sum(1-tested_units[0,:]), self.updates_size)

        # Get replacement candidates
        to_activate = np.array([random.sample(np.where(1-tested_units[i,:])[0].tolist(), k=update_size) for i in range(routing_mask.shape[0])])
        # Get candidates to discard
        to_discard = np.array([random.sample(np.where(routing_mask[i,:])[0].tolist(), k=update_size) for i in range(routing_mask.shape[0])])

        # Create the new routing mask, and update the model
        new_TR = np.array(routing_mask)
        for task in range(len(to_activate)):
            for i in range(update_size):
                new_TR[task, to_activate[task][i]] = 1
                new_TR[task, to_discard[task][i]] = 0
                
        return new_TR

    
    def train_step(self, 
                   data, 
                   gts, 
                   loss_func):
        # Run the model train step
        losses, preds = super(MR_model, self).train_step(data, 
                                                        gts, 
                                                        loss_func)
        # Update the partitions
        self.update()
        return losses, preds
    
    def freeze_partitions(self):
        for mr in self.get_MRs():
            for p in mr.parameters():
                p.requires_grad = False
    
    def get_MRs(self):
        return get_blocks(self, MRPartitioning)

    def get_MR(self, depth):
        return self.get_MRs()[depth]

    def get_routing_masks(self):
        return [elt.get_unit_mapping() for elt in self.get_MRs()]
    
    def get_routing_mask(self, depth):
        return self.get_routing_masks()[depth]
    
    def update_routing_mask(self, depth, new_mask, reset=False):
        MR_layer = self.get_MR(depth)
        MR_layer.assign_mapping(new_mask, reset=reset)

