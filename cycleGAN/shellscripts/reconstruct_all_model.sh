#!/bin/bash 

source /dtu-compute/ADNIbias/AlzPred_Oskar_Anders/gpu_AlzPred_real/bin/activate

path_to_subjects='/dtu-compute/ADNIbias/freesurfer'

SUBJECT_NAMES=$(ls /dtu-compute/ADNIbias/freesurfer/ | grep '_S_')

for f in $SUBJECT_NAMES; do 
	echo "Processing subject: $f"
    python reconstruct.py -p xy -s $f -t 1 -d /dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/oskar_implemetation/models/all_nearest_lowlearning_final -a True
    python reconstruct.py -p xz -s $f -t 1 -d /dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/oskar_implemetation/models/all_nearest_lowlearning_final -a True
    python reconstruct.py -p yz -s $f -t 1 -d /dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/oskar_implemetation/models/all_nearest_lowlearning_final -a True
    python reconstruct_all_model.py -s $f -t 1
    echo "Done with: $f"
    echo ""
done