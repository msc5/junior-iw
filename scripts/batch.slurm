#!/bin/bash

#SBATCH --job-name=vp_test                             # create a short name for your job
#SBATCH --nodes=1                                      # node count
#SBATCH --ntasks=1                                     # total number of tasks across all nodes
#SBATCH --cpus-per-task=4                              # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                                       # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1                                   # number of gpus per node
#SBATCH --time=06:00:00                                # total run time limit (HH:MM:SS)
#SBATCH --chdir=/scratch/network/msc5/code/junior-iw/
#SBATCH --mail-type=all                                # send mail when job begins, ends, or fails
#SBATCH --mail-user=msc5@princeton.edu

# module purge
# module load anaconda3/2021.5
# conda activate torch-env

for DATASET in Stocks GeneratedSins GeneratedNoise
do
	for LAYERS in 1 2 3
	do
		# bash scripts/train_model.sh LSTM $DATASET $LAYERS
		bash scripts/test_model.sh LSTM $DATASET $LAYERS
	done
done

# bash scripts/test_model.sh ConvLSTM KTH 1
# bash scripts/test_model.sh ConvLSTM KTH 2
# bash scripts/test_model.sh ConvLSTM KTH 3
#
# bash scripts/test_model.sh ConvLSTM BAIR 1
# bash scripts/test_model.sh ConvLSTM BAIR 2
# bash scripts/test_model.sh ConvLSTM BAIR 3
