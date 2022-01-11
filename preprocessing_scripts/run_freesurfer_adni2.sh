#!/bin/bash

cd /dtu-compute/ADNIbias/freesurfer_ADNI1/

source FSstable71

SUBJECT_NAMES=$(ls /dtu-compute/ADNIbias/freesurfer_ADNI2/ | grep '_S_')

for f in $SUBJECT_NAMES; do 
	echo "Processing subject: $f"
	#mri_vol2vol --reg $FREESURFER_HOME/average/mni152.register.dat --mov /dtu-compute/ADNIbias/freesurfer_ADNI1/fsaverage/mri/mni305.cor.mgz --targ /dtu-compute/ADNIbias/freesurfer_ADNI1/002_S_0295/mri/norm.mgz --inv --o 002_S_0295/norm_mni305_test_2.nii --s 002_S_0295
	mri_vol2vol --mov /dtu-compute/ADNIbias/freesurfer_ADNI2/$f/mri/norm.mgz --targ $FREESURFER_HOME/average/mni305.cor.mgz --xfm /dtu-compute/ADNIbias/freesurfer_ADNI2/$f/mri/transforms/talairach.xfm --o /dtu-compute/ADNIbias/freesurfer_ADNI2/$f/norm_mni305.mgz
done
