#!/bin/bash


#### Set-up ressources #####

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=10:00:00

# set name of job
#SBATCH --job-name=nodepint

# set number of GPUs
#SBATCH --gres=gpu:4

# mail alert at start, end and abortion of execution
##SBATCH --mail-type=ALL

# send mail to this address
##SBATCH --mail-user=gb21553@bristol.ac.uk

# Set the folder for output
#SBATCH --output ./scripts/reports/%j.out



## Some commands to quickly try this on JADE
# sbatch scripts/jade.sh  # Submit the script to JADE
# sacct -u rrn27-wwp02    # Monitor my jobs


#### run the application #####

## Load the tensorflow GPU container
# /jmain02/apps/docker/tensorflow -c

## Activate Conda (Make sure dependencies are in-there)
# module load tensorflow/2.9.1
# module load python/anaconda3
# source activate base
# conda activate jaxenv

## Run Python script
python3 ./scripts/jax_test.py
