#!/bin/bash -l

#SBATCH --job-name=Encoder_Training
#SBATCH --output=/home/ljlozenski/Desktop/USCT_Inversion/Only_SOS/encd.log
#SBATCH --error=/home/ljlozenski/Desktop/USCT_Inversion/Only_SOS/encd.err
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32768
#SBATCH --partition=volta-x86
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --mail-user=ljlozenski@lanl.gov
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

cd /home/ljlozenski/Desktop/USCT_Inversion/Only_SOS/
source /home/ljlozenski/.bashrc
source activate pytorch

#python -u random_encoding_train.py -m FCN4_Deep_Resize_2 --up_mode nearest --resume models/fcn/checkpoint.pth
python -u end2end_encoding_train.py -m FCN4_Deep_Resize_Enc --up_mode nearest  --resume models/fcn/checkpoint.pth
