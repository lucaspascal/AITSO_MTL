# Usage
All the baselines can be launched with the training script [train.py](train.py). One can choose the desired baseline with these three key arguments:
* '--method': used to choose different multi-task baselines (MTL, MGDA, GradNorm, MR, PCGrad)
* '--per_batch_step': used to choose if task should be optimized altogether (MTL-SUS) or separately (MTL-IUS/MTL-IO/MR). Set to True for MTL-SUS/GradNorm/PCGrad.
* '--one_optim_per_task': if 'per_batch_step' is True, used to choose whether to use MTL-IO (True) or MTL-IUS (False). Only supported for MTL and MR methods.

# Reproduce paper results
We provide command lines to reproduce the results obtained in the paper benchmark:

<!--sec-->
MTL-SUS:

    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method MTL --per_batch_step True --learning_rate 0.001
MTL-IUS:

    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method MTL --learning_rate 0.0005
MTL-IO:
  
    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method MTL --one_optim_per_task True --learning_rate 0.001
GradNorm:
  
    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method Gradnorm --per_batch_step True --learning_rate 0.001
PCGrad:
  
    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method PCGrad --per_batch_step True --learning_rate 0.001
Maximum Roaming:
  
    $ python train.py --name <choose name> --dataroot <data location> --checkpoint_path <save location> --method MR --learning_rate 0.001 --sigma 0.8 --updates_end 10000
    

<!--endsec-->
