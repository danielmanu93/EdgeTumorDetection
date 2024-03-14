#!/bin/bash -l

#SBATCH --job-name=Supervised_Training
#SBATCH --output=/vast/home/dmanu/Desktop/Ultra_sound/test.log
#SBATCH --error=/vast/home/dmanu/Desktop/Ultra_sound/test.err
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32000
#SBATCH --partition=volta-x86
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
source /vast/home/dmanu/.bashrc

python -u encoding_images.py
#python -u task_based_images.py
#python -u visualizeOutputs.py
#python -u classifier_images.py
