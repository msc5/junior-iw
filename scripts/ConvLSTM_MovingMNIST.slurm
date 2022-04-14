#!/bin/bash

#SBATCH --job-name=vp_mmnist       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=03:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=1,2              # job array 
#SBATCH --chdir=/scratch/network/msc5/code/junior-iw/
#SBATCH --mail-type=all        # send mail when job begins, ends, or fails
#SBATCH --mail-user=msc5@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
module load anaconda3/2021.5
conda activate torch-env

python -m src train ConvLSTM MovingMNIST \
	--task_id $SLURM_ARRAY_TASK_ID \
	--num_layers $SLURM_ARRAY_TASK_ID \
	--max_epochs 3 \
