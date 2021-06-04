import torch
import torch.nn as nn
import torch.nn.functional as F



class NYUv2Loss(nn.Module):
    """
    NYU loss module.
    """
    def __init__(self, task_groups, opt):
        super(NYUv2Loss, self).__init__()
        seg_loss_type = opt.seg_loss.upper()
        if seg_loss_type=='BCE':
            seg_loss = mean_BCE_loss if opt.seg_loss_reduction=='mean' else sum_BCE_loss
        seg_loss = mean_BCE_loss if seg_loss_type=='BCE' else nll_loss
        self.task_losses = {'segmentation': seg_loss,
                            'depth': l1_loss,
                            'normals': normal_loss,
                           }
        self.task_groups = task_groups
        self.idp_seg_tasks = opt.idp_seg_tasks and seg_loss_type=='BCE'
        self.seg_loss_reduction = opt.seg_loss_reduction
                
    def forward(self, preds, gts, group=None):
        # If specified, return loss for one given task group
        if group!=None:
            pred = preds
            # For independent segmentation tasks
            if self.idp_seg_tasks:
                # Get the correct segmentation gt
                if self.task_groups[group]['type'] == 'segmentation':
                    gt = gts[0][:,group,None,:,:]
                # Or other gts
                else:
                    gt = gts[group - gts[0].shape[1] + 1]
            # Otherwise, preds and gts have the same dimensions
            else:
                gt = gts[group]
            # Gets the correct loss
            task_loss = self.task_losses[self.task_groups[group]['type']]
            return task_loss(pred, gt)
        
        # Else compute losses for every group
        else:
            losses = []
            for k in range(len(preds)):
                pred = preds[k]
                # For independent segmentation tasks
                if self.idp_seg_tasks:
                    # Get the correct segmentation gt
                    if self.task_groups[k]['type'] == 'segmentation':
                        gt = gts[0][:,k,None,:,:]
                    # Or other gts
                    else:
                        gt = gts[k - gts[0].shape[1] + 1]
                
                # Otherwise, preds and gts have the same dimensions
                else:
                    gt = gts[k]
                task_loss = self.task_losses[self.task_groups[k]['type']]
                losses.append(task_loss(pred, gt))
            return torch.stack(losses)
            
        
                

def mean_BCE_loss(pred, gt):
    """
    BCE Segmentation loss, mean reduction.
    """
    loss = F.binary_cross_entropy(pred, gt, reduction='mean')
    return loss


def sum_BCE_loss(pred, gt):
    """
    BCE Segmentation loss, sum reduction.
    """
    loss = F.binary_cross_entropy(pred, gt, reduction='sum')
    return loss


def nll_loss(pred, gt):
    """
    NLL Segmentation loss.
    """
    argmax_gt = torch.max(gt, dim=1)[1]
    loss = F.nll_loss(pred, argmax_gt, ignore_index=-1)/5.
    return loss


def normal_loss(pred, gt):
    """
    Normal estimation loss.
    """
    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(gt, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(pred.device)
    # normal loss: dot product
    loss = 1 - torch.sum((pred * gt) * binary_mask) / torch.nonzero(binary_mask).size(0)
    return loss
    
    
def l1_loss(pred, gt):
    """
    Depth estimation loss.
    """
    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(gt, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(pred.device)
    # depth loss: l1 norm
    loss = torch.sum(torch.abs(pred - gt) * binary_mask) / torch.nonzero(binary_mask).size(0)

    return loss