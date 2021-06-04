import os
import torch
import numpy as np
import argparse
import utils
from dataset import NYUv2
from losses import NYUv2Loss
from metrics import NYUv2Metrics
from saver import Saver

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='test', type=str, help='model name')
parser.add_argument('--method', required=True, type=str, help='which model to use (MTL, PCGrad, gradnorm, MR)')

parser.add_argument('--one_optim_per_task', default=False, type=bool, help='use one optimizer for all tasks or one per task')
parser.add_argument('--per_batch_step', default=False, type=bool, help='optimize tasks altogether or one by one')
parser.add_argument('--idp_seg_tasks', default=False, type=bool, help='learn segmentation tasks altogether or separately')
parser.add_argument('--trace_param', default=False, type=bool, help='whether or not to trace parameters covered distance (slower)')

parser.add_argument('--dataroot', default='/data/nyuv2/')
parser.add_argument('--checkpoint_path', default='saved_models/nyuv2/')
parser.add_argument('--recover', default=False, type=bool, help='recover from a checkpoint')
parser.add_argument('--reco_type', default='last_checkpoint', type=str, help='which type of recovery (last_checkpoint or best_error)')
parser.add_argument('--reco_name', default=None, type=str, help='model to recover from')

parser.add_argument('--workers', default=6, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--optimizer', default='adam', type=str, help='adam or SGD')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--eval_interval', default=500, type=int, help='interval between two validations (in terms of iterations)')
parser.add_argument('--total_epoch', default=500, type=int, help='number of training epochs to proceed')
parser.add_argument('--seg_loss', default='bce', type=str, help='segmentation loss to use for segmentation tasks')
parser.add_argument('--seg_loss_reduction', default='mean', type=str, help='segmentation loss reduction method')
parser.add_argument('--seed', default=None, type=int, help='random seed')

parser.add_argument('--alpha', default=1.5, type=float, help='GradNorm hyperparameter')
parser.add_argument('--sigma', default=0.8, type=float, help='MR sharing ratio')
parser.add_argument('--updates_size', default=1, type=int, help='MR updates size')
parser.add_argument('--updates_end', default=10000, type=int, help='Time (in iteration) at which MR process should finish')
parser.add_argument('--update_ratio', default=1., type=float, help='Ratio of the MR update process to conduce')
opt = parser.parse_args()

# Seed
if opt.seed:
    torch.manual_seed(opt.seed)

# Defines input tasks
task_groups, n_groups, n_tasks = utils.create_task_groups(opt)

# Saving settings
model_dir = os.path.join(opt.checkpoint_path, opt.name)
os.mkdir(model_dir) if not os.path.isdir(model_dir) else None
saver = Saver(model_dir, args=opt)

# Define model and optimiser
gpu = utils.check_gpu()
device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
model = utils.select_model(opt, task_groups).to(device)

# Loss and metrics
loss_func = NYUv2Loss(task_groups, opt)
train_metrics = NYUv2Metrics(task_groups, opt)
val_metrics = NYUv2Metrics(task_groups, opt)
logger = utils.logger(train_metrics.metrics_names)
saver.add_metrics(train_metrics.metrics_names)

# Recover weights, if required
model.initialize(opt, device, model_dir, saver)

# Create datasets and loaders
dataset_path = opt.dataroot
use_bg = opt.seg_loss.upper()=='NLL'
train_data = NYUv2(dataset_path, 
                   train=True, 
                   download=True,
                   bg_class=use_bg)
val_data = NYUv2(dataset_path, 
                 train=False, 
                 bg_class=use_bg)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    shuffle=True,
    drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataset=val_data,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    shuffle=False)


# Few parameters
total_epoch = opt.total_epoch
nb_train_batches = len(train_loader)
nb_val_batches = len(val_loader)
train_avg_losses = np.zeros(n_groups, dtype=np.float32)


# Iterations
while model.n_epoch < total_epoch:
    ####################
    ##### Training #####
    ####################
    model.train()
    train_dataset = iter(train_loader)
    for train_k in range(nb_train_batches):
        # Data loading
        train_data, train_gts = train_dataset.next()
        train_data = train_data.to(device)
        train_gts = [elt.to(device) for elt in train_gts]

        # Train step
        task_losses, train_preds = model.train_step(train_data, train_gts, loss_func)
        
        # Scoring
        train_avg_losses += task_losses.cpu().numpy()
        train_metrics.incr(train_preds, train_gts)
        
        # Logging
        print('Epoch {}, iter {}/{}, losses ({:.2f}, {:.2f}, {:.2f})'.format(model.n_epoch, train_k, nb_train_batches, task_losses[0], task_losses[1], task_losses[2]), end=' '*50+'\r')
        

        ######################
        ##### Validation #####
        ######################
        if model.n_iter%opt.eval_interval == 0:
            val_avg_losses = np.zeros(n_groups, dtype=np.float32)
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
            train_avg_losses /= opt.eval_interval
            train_results = train_metrics.result()
            val_results = val_metrics.result()
            logger.log(model.n_epoch, 
                       model.n_iter, 
                       train_results, 
                       val_results, 
                       train_avg_losses.sum(), 
                       val_avg_losses.sum())
            saver.log(model, 
                      task_groups, 
                      model.n_epoch, 
                      model.n_iter, 
                      train_results, 
                      val_results, 
                      train_avg_losses, 
                      val_avg_losses,
                      model.optimizer)
            train_metrics.reset()
            val_metrics.reset()
            model.train()

    
    # Update epoch and LR
    model.n_epoch += 1

