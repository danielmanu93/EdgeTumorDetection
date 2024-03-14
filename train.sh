#!/bin/bash -l

#SBATCH --job-name=Supervised_Training
#SBATCH --output=/vast/home/dmanu/Desktop/Ultra_sound/train.log
#SBATCH --error=/vast/home/dmanu/Desktop/Ultra_sound/train.err
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=300000
#SBATCH --partition=clx-volta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --mail-user=dmanu@lanl.gov
#SBATCH --chdir=/vast/home/dmanu/Desktop/Ultra_sound
#SBATCH --mail-type=ALL

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $SLURM_JOB_NAME"
echo "Job ID : $SLURM_JOB_ID" 
echo "=========================================================="
cat /etc/redhat-release

MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
MASTERPORT=6000

echo "Master Node: $MASTER"
echo "Slave Node(s): $SLAVES"

cd /vast/home/dmanu/Desktop/Ultra_sound/USCT_InversionNet

#python -u forward_run.py
# 
#python -u train.py -m FCN4_Deep_Resize_2 --up_mode nearest #--resume models/fcn/checkpoint.pth #--train-anno data2.txt  ### No source encoding 
srun python -u end2end_encoding_train.py -n fcn_trained_encoder -o /vast/home/dmanu/Desktop/Ultra_sound/checkpoints/ -s run2 -m FCN4_Deep_Resize_Enc --up_mode nearest --tensorboard -te True #--resume models/fcn/train_checkpoint.pth  ### Trained source encoding
#python -u train_task_based.py -m FCN4_Deep_Resize_Enc --up_mode nearest -n fcn_task -o /vast/home/dmanu/Desktop/Ultra_sound/checkpoints/ -s run1 --tensorboard  #--resume models/fcn/genn_checkpoint.pth #### Task based training
