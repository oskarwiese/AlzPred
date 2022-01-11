
from re import S
import torch
import os
import argparse
import glob
from torchvision.datasets import folder
import config
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import nibabel as nib
from skimage import io
from nipy import load_image
from torchvision import transforms
import torchvision.datasets as datasets 
import torch.optim as optim
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_MRI import Generator
from torch.utils.data import DataLoader
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

subjects = ["082_S_0469", "136_S_0196"]
    
    if tesla == 3:
        path = f'/dtu-compute/ADNIbias/freesurfer/{sub}/norm_mni305.mgz'
    else:
        path = f'/dtu-compute/ADNIbias/freesurfer_ADNI1/{sub}/norm_mni305.mgz'
       
    img = load_image(path)
    img_xy = np.array(img[:,:,z]).astype(np.uint8)
            if img_xy.sum() != 0:
                imgs_xy.append((img_xy,z))
                io.imsave(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/xy/{tesla}/{tesla}/{str(z).zfill(3)}_{sub}_xy_z_{z}_{tesla}T.png', img_xy, check_contrast=False)