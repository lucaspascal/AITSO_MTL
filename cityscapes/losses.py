import torch
import torch.nn as nn
import torch.nn.functional as F



class CityscapesLoss(nn.Module):
    """
    Cityscapes loss module.
    """
    def __init__(self, task_groups, opt):
        super(CityscapesLoss, self).__init__()
        seg_loss = BCE_loss
        self.task_losses = {'segmentation': seg_loss,
                           }
        self.task_groups = task_groups
                
    def forward(self, preds, gts, group=None):
        # If specified, return loss for one given task group
        if group!=None:
            pred = preds
            gt = gts[:,group,None,:,:]
            task_loss = self.task_losses[self.task_groups[group]['type']]
            return task_loss(pred, gt)
        
        # Else compute losses for every group
        else:
            losses = []
            for k in range(len(preds)):
                pred = preds[k]
                gt = gts[:,k,None,:,:]
                task_loss = self.task_losses[self.task_groups[k]['type']]
                losses.append(task_loss(pred, gt))
            return torch.stack(losses)
            
        
                

def BCE_loss(pred, gt):
    """
    BCE Segmentation loss.
    """
    loss = F.binary_cross_entropy(pred, gt, reduction='mean')
    return loss

