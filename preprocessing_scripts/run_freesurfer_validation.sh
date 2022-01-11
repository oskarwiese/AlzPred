#!/bin/bash

cd /dtu-compute/ADNIbias/freesurfer_ADNI1/

source FSstable71

cd /dtu-compute/ADNIbias/freesurfer/


SUBJECT_NAMES=$(ls /dtu-compute/ADNIbias/freesurfer/ | grep '_S_')

for f in $SUBJECT_NAMES; do 
	echo "Processing subject: $f"
	mri_vol2vol --mov /dtu-compute/ADNIbias/freesurfer/$f/mri/norm.mgz --targ $FREESURFER_HOME/average/mni305.cor.mgz --xfm /dtu-compute/ADNIbias/freesurfer/$f/mri/transforms/talairach.xfm --o /dtu-compute/ADNIbias/freesurfer/$f/norm_mni305.mgz
done


