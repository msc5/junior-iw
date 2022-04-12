#!/bin/bash

#SBATCH --job-name=video_pred    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=msc5@princeton.edu

module purge
module load anaconda3/2021.5
conda activate torch-env

# python -m src train ConvLSTM MovingMNIST --num_layers 2
# python -m src train ConvLSTM KTH --num_layers 4
# python -m src train ConvLSTM KTH --num_layers 1
# python -m src train ConvLSTM KTH --num_layers 2 --kth_classes walking
# python -m src train ConvLSTM KTH --num_layers 4 --kth_classes walking
python -m src train ConvLSTM BAIR --num_layers 2
