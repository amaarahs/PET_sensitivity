#!/bin/bash -l
# Batch script to run a serial job under SGE.

#$ -l gpu=1
#$ -l h_rt=1:0:0
#$ -l mem=16G
#$ -l tmpfs=15G
#$ -N Create_data

# This is to send reminder emails to yourself when your job begins and ends
#$ -m be
#$ -M ucapas9@ucl.ac.uk

# Set the working directory to somewhere in your scratch space.  
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
#$ -wd /home/ucapas9/Scratch/

# Your work should be done in $TMPDIR
cd $TMPDIR

# Copy PET directory to $TMPDIR
cp -r /home/ucapas9/Scratch/PET_sensitivity/ $TMPDIR/

# Load required modules
module -f unload compilers gcc-libs openblas cuda cudnn python3
module load beta-modules
module load compilers/gnu/10.2.0
module load gcc-libs/10.2.0
module unload openblas
module load openblas/0.3.13-serial/gnu-10.2.0 
module load python3/3.9-gnu-10.2.0
module load cuda/11.3.1/gnu-10.2.0

# module load cudnn/8.2.1.32/cuda-11.3
module load pytorch/1.11.0/gpu
# Batch script to run a serial

# Activate virtual environment
source /home/ucapas9/venv/bin/activate

# Set the number of OpenMP threads
#export OMP_NUM_THREADS=12
#echo "Using OpenBLAS with OpenMP threads: $OMP_NUM_THREADS"

python PET_sensitivity/scripts/create_data.py -n 10000 -s "/home/ucapas9/Scratch/output/"

# Copy results back to scratch directory 
mkdir -p /home/ucapas9/Scratch/output/
cp -r $TMPDIR/PET_sensitivity/data/training_data /home/ucapas9/Scratch/output/
