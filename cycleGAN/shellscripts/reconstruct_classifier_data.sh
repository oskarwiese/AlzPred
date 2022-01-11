#!/bin/bash 

source /dtu-compute/ADNIbias/AlzPred_Oskar_Anders/gpu_AlzPred_real/bin/activate

path_to_subjects='/dtu-compute/ADNIbias/freesurfer_ADNI2/'

SUBJECT_NAMES=$(ls /dtu-compute/ADNIbias/freesurfer_ADNI2/ | grep '_S_')

for f in $SUBJECT_NAMES; do 
	echo "Processing subject: $f"
    python reconstruct.py -p xy -s $f -t 3 -d /dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/oskar_implemetation/models/xy_nearest_lowlearning_final 
    echo "Done with: $f"
    echo ""
done