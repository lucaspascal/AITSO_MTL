import torch
import torch.nn as nn



class CelebaLoss(nn.Module):
    """
    CelebA loss module.
    """
    def __init__(self):
        super(CelebaLoss, self).__init__()
        self.task_loss = nn.BCELoss(reduction='none')
                
    def forward(self, logits, gts, active_task=None):
        # If specified, return loss for one given task group
        if active_task!=None:
            log = logits
            gt = gts[active_task]
            return self.task_loss(log, gt).mean(0).sum()
        # Else compute losses for every group
        else:
            losses = []
            for k in range(len(gts)):
                log = logits[k]
                gt = gts[k]
                losses.append(self.task_loss(log, gt).mean(0).sum())
            return torch.stack(losses)
