#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J oskarCNN
### -- ask for number of cores (default: 1) -- 
#BSUB -n 6 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- Select resoruces: 1 gpu in exclusive mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=42GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 42GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Error_%J.err 

# here follow the commands you want to execute 

source /dtu-compute/ADNIbias/AlzPred_Oskar_Anders/gpu_AlzPred_real/bin/activate

module load cuda/11.0
python /dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/src/train_classifier.py
