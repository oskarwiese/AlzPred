#!/bin/bash

declare -a StringArray=('002_S_0295' '002_S_0619' '002_S_0685' '003_S_0907' '003_S_0931' '003_S_0981' '003_S_1021' '003_S_6014' '003_S_6067' '003_S_6092' '003_S_6264' '003_S_6833' '005_S_0221' '005_S_0814' '005_S_0929' '006_S_0547' '006_S_0653' '007_S_1248' '007_S_1304' '007_S_1339' '009_S_1334' '009_S_1354')

for val in ${StringArray[@]}; do
    echo $val
    cd /dtu-compute/ADNIbias/freesurfer_ADNI1/$val/
    cp norm_mni305.mgz norm_mni305_$val.mgz
    mv norm_mni305_$val.mgz /dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/imgs_non_cycleGAN/
done
