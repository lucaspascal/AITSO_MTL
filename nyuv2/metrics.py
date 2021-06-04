import numpy as np
import torch

CLASS_NB = 13    
SMOOTH = 1e-7      


class Metric():
    """
    Basic metric accumulator of chosen size.
    """
    def __init__(self, shape, ignore=-1.):
        self.acc = np.zeros(shape, dtype=np.float32)
        self.count = np.zeros(shape, dtype=np.int32)
        self.ignore = -1.
        self.linked_metrics = None
        self.link_type = None
        
    def link(self, metrics, link_type='mean'):
        self.linked_metrics = metrics
        self.link_type = link_type
        
    def incr(self, value):
        value = np.array([value])
        for k in range(value.size):
            if value[k] != self.ignore:
                self.acc[k] += value[k]
                self.count[k] += 1
                
    def result(self, reduce='mean'):
        if self.linked_metrics != None:
            if self.link_type == 'mean':
                res = np.mean([elt.result() for elt in self.linked_metrics])
        else:
            res = self.acc / (self.count + 1e-7)
            if reduce=='mean':
                res = np.mean(res)
            elif reduce=='sum':
                res = np.sum(res)
        return res
    
    def reset(self):
        shape = self.acc.shape
        self.acc = np.zeros(shape, dtype=np.float32)
        self.count = np.zeros(shape, dtype=np.int32)


class NYUv2Metrics():
    """
    NYU metric accumulator.
    """
    def __init__(self, task_groups, opt):
        self.task_groups = task_groups
        self.n_tasks = len(task_groups)
        self.n_classes = sum([len(elt) for elt in task_groups])
        self.seg_loss = opt.seg_loss.upper()
        self.idp_seg_tasks = opt.idp_seg_tasks and self.seg_loss=='BCE'
        self.metrics = []
        self.metrics_names = []
        
        # Adds all NYU metrics
        for k in range(CLASS_NB):
            self.add_metric('IoU_{}'.format(k))
        self.add_metric('mIoU')
        self.link_metric('mIoU', ['IoU_{}'.format(k) for k in range(CLASS_NB)])
        self.add_metric('Pix_Acc')
        self.add_metric('Abs_Error')
        self.add_metric('Rel_Error')
        self.add_metric('Mean_Error')
        self.add_metric('Median_Error')
        self.add_metric('<11.25')
        self.add_metric('<22.5')
        self.add_metric('<30')
        
        # Set metrics to 0
        self.reset()
    
    def link_metric(self, source_metric, target_metrics, link_type='mean'):
        self.metrics[self.metrics_names.index(source_metric)].link([self.metrics[self.metrics_names.index(elt)] for elt in target_metrics], link_type)
        
    def add_metric(self, name, shape=1):
        self.metrics.append(Metric(shape))
        self.metrics_names.append(name)
        
    def incr_metric(self, name, value):
        self.metrics[self.metrics_names.index(name)].incr(value)

    def reset(self):
        self.n_samples = 0
        for met in self.metrics:
            met.reset()
                
    def incr(self, preds, gts):
        n_class = CLASS_NB if self.seg_loss=='BCE' else CLASS_NB+1
        seg_preds = [None]*n_class
        for (pred, task) in zip(preds, range(self.n_tasks)):
            if self.task_groups[task]['type'] == 'depth':
                abs_err, rel_err = depth_error(pred, gts[1])
                self.incr_metric('Abs_Error', abs_err.item())
                self.incr_metric('Rel_Error', rel_err.item())
            elif self.task_groups[task]['type'] == 'normals':
                mean_err, med_err, inf_1, inf_2, inf_3 = normal_error(pred, gts[2])
                self.incr_metric('Mean_Error', mean_err)
                self.incr_metric('Median_Error', med_err)
                self.incr_metric('<11.25', inf_1)
                self.incr_metric('<22.5', inf_2)
                self.incr_metric('<30', inf_3)
            elif self.task_groups[task]['type'] == 'segmentation':
                if self.idp_seg_tasks:
                    seg_preds[self.task_groups[task]['ids'][0]] = pred[:,0,:,:]
                else:
                    for task_id in self.task_groups[task]['ids']:
                        seg_preds[task_id] = pred[:,task_id,:,:]
                    
        # Create unified segmentation tensors
        seg_preds = torch.stack(seg_preds, dim=1)
        seg_gts = gts[0].type(torch.int8)
        
        # Scores segmentations
        ious, pix_acc = compute_seg_metrics(seg_preds, seg_gts, self.seg_loss)

        for k in range(len(ious)):
            self.incr_metric('IoU_{}'.format(k), ious[k])
        self.incr_metric('Pix_Acc'.format(k), pix_acc)
                
        self.n_samples += 1
                
        
    def result(self):
        res = []
        for met in self.metrics:
            res.append(met.result())
        return res
    

        
def compute_seg_metrics(preds, gts, loss_type):
    """
    Computes segmentation metrics depending if we proceed multi-class 
    binary segmentation (BCE) or basic semantic segmentation (NLL)
    """
    if loss_type=='BCE':
        # Generates binary predictions from logits
        bin_preds = (preds>=0.5).type(torch.int8)
    
        # Create argmax prediction and gt maps (with background accounted)
        argmax_preds = (torch.max(preds, dim=1, keepdim=True)[1] + 1).long()
        bg_preds = (bin_preds.sum(dim=1, keepdim=True)>0.).long()
        argmax_preds *= bg_preds
    
        argmax_gts = (torch.max(gts, dim=1, keepdim=True)[1] + 1).long()
        bg_gts = (gts.sum(dim=1, keepdim=True)>0.).long()
        argmax_gts *= bg_gts
    
        # Computes Pixel Accuracy
        pix_acc = compute_pix_acc(argmax_preds, argmax_gts)
    
        # Compute ious on independent task maps
        ious = compute_ious(bin_preds, gts)
        
    elif loss_type=='NLL':
        # Create argmax prediction and gt maps (with background accounted)
        argmax_preds = (torch.max(preds[:,1:, :, :], dim=1, keepdim=True)[1]).long()
        argmax_gts = (torch.max(gts[:,1:, :, :], dim=1, keepdim=True)[1]).long()
        
        # Computes Pixel Accuracy
        pix_acc = compute_pix_acc(argmax_preds, argmax_gts)
        ious = compute_argmax_ious(argmax_preds, argmax_gts)

    return ious, pix_acc

def compute_ious(bin_pred_maps, gt_maps):
    """
    Computes ious for multi-class binary segmentation (BCE)
    """
    intersections = (bin_pred_maps & gt_maps).float().sum((2,3))
    unions = (bin_pred_maps | gt_maps).float().sum((2,3))
    class_ious = ((intersections) / (unions + SMOOTH)).transpose(0,1)
    
    true_classes = ((gt_maps.transpose(0,1).sum((2,3))) > 0.).type(torch.float32)
    filtered_ious = class_ious.sum(1) / (true_classes.sum(1) + SMOOTH)
    filtered_ious[true_classes.sum(1)==0] = -1.
    return filtered_ious
    
def compute_argmax_ious(pred, gt):
    """
    Computes ious for basic semantic segmentation (NLL)
    """
    device = pred.device
    batch_size = pred.size(0)
    true_class = 0
    first_switch = True
    ious = []
    for j in range(CLASS_NB):
        pred_mask = (pred == j).type(torch.FloatTensor).to(device)
        gt_mask = (gt == j).type(torch.FloatTensor).to(device)
        mask_comb = pred_mask + gt_mask
        union     = torch.sum((mask_comb > 0).type(torch.FloatTensor), dim=(1,2)).to(device)
        intsec    = torch.sum((mask_comb > 1).type(torch.FloatTensor), dim=(1,2)).to(device)
        class_iou = intsec / (union + 1e-7)
        true_class = (gt_mask.sum((1,2)) > 0).type(torch.float32)
        if true_class.sum() == 0.:
            batch_class_iou = torch.tensor(-1, dtype=torch.float32).to(device)
        else:
            batch_class_iou = class_iou.sum() / true_class.sum()
        ious.append(batch_class_iou)
    ious = torch.stack(ious)
    true_ious = ious >= 0.
    miou = ious[true_ious].mean()
    return ious


def compute_pix_acc(argmax_preds, argmax_gts):
    """
    Computes pixel accuracy.
    """
    return (argmax_preds == argmax_gts).float().mean()


def depth_error(pred, gt):
    """
    Computes depth error between pred and gt.
    """
    binary_mask = (torch.sum(gt, dim=1) != 0).unsqueeze(1)
    x_pred_true = pred.masked_select(binary_mask)
    x_output_true = gt.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)

def normal_error(pred, gt):
    """
    Computes normal error between pred and gt.
    """
    binary_mask = (torch.sum(gt, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(pred * gt, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)
