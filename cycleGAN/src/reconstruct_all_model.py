
import numpy as np
import torch
import argparse
import config
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import nibabel as nib
from skimage import io
from nipy import load_image
import warnings
import nibabel
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from nipy import load_image
import argparse


def dat_recon(subject, tesla):


    tmp1 =  load_image(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/all_xy_{tesla}_{subject}.nii')
    tmp2 =  load_image(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/all_xz_{tesla}_{subject}.nii')
    tmp3 =  load_image(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/all_yz_{tesla}_{subject}.nii')
    
    tmp1 = np.array(tmp1[:,:,:])
    tmp2 = np.array(tmp2[:,:,:])
    tmp3 = np.array(tmp3[:,:,:])

    mean = np.mean((tmp1,tmp2, tmp3), axis = 0)

    transformation = np.array([[  -1.,    0.,    0.,  128.],
                               [   0.,    0.,    1., -128.],
                               [   0.,   -1.,    0.,  128.],
                               [   0.,    0.,    0.,    1.]])   # Affine transformation matrix    

    img3d = nib.Nifti1Image(mean, affine=transformation)
    nib.save(img3d, f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/reconstructed/all_final_{tesla}_{subject}.nii') #TODO create better naming convention

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--subject', type=str, help="Example of subject: 002_S_4171")
    parser.add_argument("-t", '--tesla', type=int, help="Either 1.5 or 3 tesla, note that if you choose 3T a 1.5T version will be generated")
    args = parser.parse_args()

    if not args.subject:
        print('No subject chosen, so subject default is: 002_S_4171 \n')
        args.subject = '002_S_4171'
    if not args.tesla:
        print('No field strength chosen, default is: 3 \n')
        args.tesla = 3
    if args.tesla != 3:
        args.tesla = 1.5

    subject, tesla = args.subject, args.tesla

    dat_recon(subject, tesla)
    print('Done w dat recon \n')

#path = f'/dtu-compute/ADNIbias/freesurfer_ADNI1/002_S_0295/norm_mni305.mgz'
#img = load_image(path)
#print(img.affine)
#
#path = f'/dtu-compute/ADNIbias/freesurfer_ADNI1/003_S_0931/norm_mni305.mgz'
#img = load_image(path)
#print(img.affine)
#
#path = f'/dtu-compute/ADNIbias/freesurfer_ADNI1/006_S_0653/norm_mni305.mgz'
#img = load_image(path)
#print(img.affine)
#
#path = f'/dtu-compute/ADNIbias/freesurfer_ADNI1/007_S_0293/norm_mni305.mgz'
#img = load_image(path)
#print(img.affine)