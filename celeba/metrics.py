import numpy as np

class CelebaMetrics():
    """
    CelebA metric accumulator.
    """
    def __init__(self, task_groups):
        self.task_groups = task_groups
        self.n_tasks = len(task_groups)
        self.n_classes = sum([len(elt) for elt in task_groups])
        self.metrics = ['precision', 'recall', 'fscore', 'avg_accuracy', 'avg_error']
        self.reset()
        
    def reset(self):
        self.n_samples = 0
        self.pred = np.zeros(self.n_classes, dtype=np.float32)
        self.to_pred = np.zeros(self.n_classes, dtype=np.float32)
        self.well_pred = np.zeros(self.n_classes, dtype=np.float32)
        self.accuracies = np.zeros(self.n_classes, dtype=np.float32)
        
    def incr(self, preds, gts):
        for (pred, gt, task) in zip(preds, gts, range(self.n_tasks)):
            self.incr_task(pred.cpu().numpy(), gt.cpu().numpy(), task)
        self.n_samples += 1
                
    def incr_task(self, pred, gt, task):
        classes = self.task_groups[task]
        self.pred[classes] += pred.sum(axis=0)
        self.to_pred[classes] += gt.sum(axis=0)
        self.well_pred[classes] += (pred*gt).sum(axis=0)
        self.accuracies[classes] += (pred==gt).mean(axis=0)
        
    def result(self):
        precisions = self.well_pred / (self.pred + 1e-7)
        recalls = self.well_pred / (self.to_pred + 1e-7)
        fscores = 2*precisions*recalls / (precisions + recalls + 1e-7)
        accuracies = self.accuracies / self.n_samples
        errors = 1. - accuracies
        
        precision = precisions.mean()
        recall = recalls.mean()
        fscore = fscores.mean()
        avg_accuracy = accuracies.mean()
        avg_error = errors.mean()
        
        return precision, recall, fscore, avg_accuracy, avg_error
    
        