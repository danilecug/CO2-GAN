#!/bin/bash 
#module load gcc/4.9.1
#module load cuda/8.0
#module load cudnn/5.1
#module load python3/3.5.2
#module load tacc-singularity
#echo 'All of the modules have been loaded'
#
#echo '======================================================='
#echo '============Begin to Training the CO2GAN =============='
#echo '======================================================='
#
#image='/singularity_cache/tacc-maverick-ml-latest.simg'
#checkpoint='checkpoint_Oct_12'
#singularity exec --nv ${STOCKYARD}$image python MODULE_DCGAN_Train.py --chpt $checkpoint
checkpoint='../Model/Model_Oct_29'
#python MODULE_DCGAN_Train.py --model $checkpoint
#python MODULE_DCGAN_Test_no_dataloader.py --model $checkpoint
python MODULE_DCGAN_Test_no_dataloader.py --model $checkpoint 

