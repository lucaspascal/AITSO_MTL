import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import argparse
import torch.utils.data.sampler as sampler
import utils
from dataset import CelebaGroupedDataset
from saver import Saver
from losses import CelebaLoss
from metrics import CelebaMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='test', type=str, help='model name')
parser.add_argument('--method', required=True, type=str, help='which model to use (MTL, MGDA, PCGrad, gradnorm, MR)')
parser.add_argument('--split', default='test', type=str, help='On which split to evaluate the model')

parser.add_argument('--groups_size', type=int, default=1, help='Size of task groups. Only modify it when using MTL or MR models, otherwise leave it to 1.')
parser.add_argument('--one_optim_per_task', default=False, type=bool, help='use one optimizer for all tasks or one per task')

parser.add_argument('--trace_param', default=False, type=bool, help='whether or not to trace parameters covered distance (slower)')

parser.add_argument('--dataroot', default='/data/celeba/')
parser.add_argument('--checkpoint_path', default='/savec_models/celeba/')
parser.add_argument('--recover', default=False, type=bool, help='recover from a checkpoint')
parser.add_argument('--reco_type', default='last_checkpoint', type=str, help='which type of recovery (last_checkpoint or best_error)')
parser.add_argument('--reco_name', default=None, type=str, help='model to recover from')

parser.add_argument('--workers', default=6, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--eval_interval', default=636, type=int, help='interval between two validations')
parser.add_argument('--total_epoch', default=15, type=int, help='number of training epochs to proceed')
parser.add_argument('--seed', default=None, type=int, help='random seed')

parser.add_argument('--alpha', default=0.2, type=float, help='GradNorm hyperparameter')
parser.add_argument('--sigma', default=0.8, type=float, help='MR sharing ratio')
parser.add_argument('--updates_size', default=1, type=int, help='MR updates size')
parser.add_argument('--updates_end', default=6000, type=int, help='Time (in iteration) at which MR process should finish')
parser.add_argument('--update_ratio', default=1., type=float, help='Ratio of the MR update process to conduce')
opt = parser.parse_args()

# Defines input tasks
task_groups = np.load(os.path.join(opt.checkpoint_path, opt.name, 'task_groups.npy'))
tasks = [k for k in range(len(task_groups))]
n_tasks = len(task_groups)

# Saving settings
model_dir = os.path.join(opt.checkpoint_path, opt.name)
os.mkdir(model_dir) if not os.path.isdir(model_dir) else None
saver = Saver(model_dir, args=opt)

# Define model and optimiser
gpu = utils.check_gpu()
device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
model = utils.select_model(opt, task_groups).to(device)

# Loss and metrics
loss_func = CelebaLoss()
train_metrics = CelebaMetrics(task_groups)
val_metrics = CelebaMetrics(task_groups)
logger = utils.logger(train_metrics.metrics)
saver.add_metrics(train_metrics.metrics)

# Recover weights, if required
model.initialize(opt, device, model_dir, saver)

# Create datasets
dataset_path = opt.dataroot
val_data = CelebaGroupedDataset(dataset_path, 
                                task_groups, 
                                split=opt.split)


batch_size = opt.batch_size
val_loader = torch.utils.data.DataLoader(
    dataset=val_data,
    batch_size=batch_size,
    num_workers=opt.workers,
    shuffle=False)

# Define parameters
total_epoch = opt.total_epoch
nb_val_batches = len(val_loader)

######################
##### Validation #####
######################
val_avg_losses = np.zeros(n_tasks, dtype=np.float32)
model.eval()
val_dataset = iter(val_loader)
with torch.no_grad():
    for val_k in range(nb_val_batches):
        # Data loading
        val_data, val_gts = val_dataset.next()
        val_data = val_data.to(device)
        val_gts = [elt.to(device) for elt in val_gts]

        # Logging
        print('Eval {}/{} epoch {} iter {}'.format(val_k, nb_val_batches, model.n_epoch, model.n_iter), end=' '*50+'\r')

        # Test step
        task_losses, val_preds = model.test_step(val_data, val_gts, loss_func)

        # Scoring
        val_avg_losses += task_losses.cpu().numpy() / nb_val_batches
        val_metrics.incr(val_preds, val_gts)


# Logging
val_results = val_metrics.result()
print('='*50)
print('Avg. error:', val_results[4])
print('Fscore:', val_results[2])

