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

# Seed
if opt.seed:
    torch.manual_seed(opt.seed)

# Defines input tasks
task_groups = utils.create_task_groups(opt.groups_size)
tasks = [k for k in range(len(task_groups))]
n_tasks = len(task_groups)

# Saving settings
model_dir = os.path.join(opt.checkpoint_path, opt.name)
os.mkdir(model_dir) if not os.path.isdir(model_dir) else None
np.save(os.path.join(opt.checkpoint_path, opt.name, 'task_groups.npy'), task_groups)
saver = Saver(model_dir, args=opt)

# Define model and optimiser
gpu = utils.check_gpu()
device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
model = utils.select_model(opt, task_groups).to(device)

# Loss and metrics
loss_func = CelebaLoss()
train_metrics = CelebaMetrics(task_groups)
val_metrics = CelebaMetrics(task_groups)
test_metrics = CelebaMetrics(task_groups)
logger = utils.logger(train_metrics.metrics)
saver.add_metrics(train_metrics.metrics)

# Recover weights, if required
model.initialize(opt, device, model_dir, saver)

# Create datasets
dataset_path = opt.dataroot
train_data = CelebaGroupedDataset(dataset_path, 
                                  task_groups, 
                                  split='train')
val_data = CelebaGroupedDataset(dataset_path, 
                                task_groups, 
                                split='val')
test_data = CelebaGroupedDataset(dataset_path, 
                                task_groups, 
                                split='test')

batch_size = opt.batch_size
train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    num_workers=opt.workers,
    shuffle=True,
    drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataset=val_data,
    batch_size=batch_size,
    num_workers=opt.workers,
    shuffle=False)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    num_workers=opt.workers,
    shuffle=False)


# Few parameters
total_epoch = opt.total_epoch
nb_train_batches = len(train_loader)
nb_val_batches = len(val_loader)
nb_test_batches = len(test_loader)
train_avg_losses = np.zeros(n_tasks, dtype=np.float32)


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
        print('Epoch {}, iter {}/{}, loss {:.2f}'.format(model.n_epoch, train_k, nb_train_batches, task_losses.sum()), end=' '*50+'\r')
        

        ######################
        ##### Validation #####
        ######################
        if model.n_iter%opt.eval_interval == 0:
            ### Val set ###
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

        
            ### Test set ###
            test_avg_losses = np.zeros(n_tasks, dtype=np.float32)
            test_dataset = iter(test_loader)
            with torch.no_grad():
                for test_k in range(nb_test_batches):
                    # Data loading
                    test_data, test_gts = test_dataset.next()
                    test_data = test_data.to(device)
                    test_gts = [elt.to(device) for elt in test_gts]

                    # Logging
                    print('Test {}/{} epoch {} iter {}'.format(test_k, nb_test_batches, model.n_epoch, model.n_iter), end=' '*50+'\r')

                    # Test step
                    task_losses, test_preds = model.test_step(test_data, test_gts, loss_func)

                    # Scoring
                    test_avg_losses += task_losses.cpu().numpy() / nb_test_batches
                    test_metrics.incr(test_preds, test_gts)

        
            # Logging
            train_avg_losses /= opt.eval_interval
            train_results = train_metrics.result()
            val_results = val_metrics.result()
            test_results = test_metrics.result()
            logger.log(model.n_epoch, 
                           model.n_iter, 
                           train_results, 
                           val_results, 
                           train_avg_losses.sum(), 
                           val_avg_losses.sum())
            saver.log(model, 
                      model.n_epoch, 
                      model.n_iter, 
                      train_results, 
                      val_results, 
                      test_results, 
                      train_avg_losses.sum(), 
                      val_avg_losses.sum(), 
                      test_avg_losses.sum(), 
                      model.optimizer)
            train_metrics.reset()
            val_metrics.reset()
            test_metrics.reset()
            model.train()

    
    # Update epoch and LR
    model.n_epoch += 1

