# Usage
All the baselines can be launched with the training script [train.py](train.py). They can then be tested on the test set with [test.py](test.py). One can choose the desired baseline with these three key arguments:
* '--method': used to choose different multi-task baselines (MTL, MGDA, GradNorm, MR, PCGrad)
* '--groups_size: defines the size of task groups for the random grouping strategy applied to MTL-IUS and IO. Set to 40 to use MTL-SUS, and to 1 to use MTL-IUS/IO withouth the grouping strategy. Leave it to 1 for MGDA/MR/GradNorm/PCGrad.
* '--one_optim_per_task': if 'per_batch_step' is True, used to choose whether to use MTL-IO (True) or MTL-IUS (False). Only supported for MTL and MR methods.

# Reproduce paper results
We provide command lines to reproduce the results obtained in the paper benchmark:

<!--sec-->
MTL-SUS:

    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method MTL --groups_size 40 --learning_rate 0.001
MTL-IUS (8 groups):

    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method MTL --groups_size 5 --learning_rate 0.0005
MTL-IO (4 groups):
  
    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method MTL --groups_size 10 --one_optim_per_task True --learning_rate 0.0005
MTL-IUS (40 groups/no grouping):

    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method MTL --groups_size 1 --learning_rate 0.0001
MTL-IO (40 groups/no grouping):
  
    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method MTL --groups_size 1 --one_optim_per_task True --learning_rate 0.0001
GradNorm:
  
    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method Gradnorm --learning_rate 0.001
PCGrad:
  
    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method PCGrad --learning_rate 0.001
MGDA-UB:
  
    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method MGDA --learning_rate 0.001
Maximum Roaming:
  
    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method MR --learning_rate 0.0005 --sigma 0.8 --updates_end 3000
    

<!--endsec-->
