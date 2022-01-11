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
from reconstruct import subject_images

parser = argparse.ArgumentParser() ##
parser.add_argument("-s", '--subject', type=str,
                help="Example of subject: 002_S_4171")
parser.add_argument("-t", '--tesla', type=int,
                help="Either 1.5 or 3 tesla, note that if you choose 3T a 1.5T version will be generated")
parser.add_argument("-a", '--all', type=bool,
                help="Will the model be generating all_model slices?")
args = parser.parse_args()


if not args.subject:
    print('No subject chosen, so subject default is: 002_S_4171 \n')
    args.subject = '002_S_4171'
if not args.tesla:
    print('No field strength chosen, default is: 3 \n')
    args.tesla = 3
if args.tesla != 3:
    args.tesla = 1.5


print(f'Clearing folders...\n')
folder_paths = [
        f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/xy/{args.tesla}/{args.tesla}/',
        f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/xz/{args.tesla}/{args.tesla}/',
        f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/reconstruct/yz/{args.tesla}/{args.tesla}/']

for folder_path in tqdm(folder_paths):
     import shutil
     shutil.rmtree(folder_path)
     os.mkdir(folder_path )

xy, xz, yz = subject_images(args.subject, tesla = args.tesla) # 002_S_4171 is first subject


idx_black_xy = [min([val[1] for val in xy]), max([val[1] for val in  xy])]
idx_black_xz = [min([val[1] for val in xz]), max([val[1] for val in  xz])]
idx_black_yz = [min([val[1] for val in yz]), max([val[1] for val in  yz])]

xy_slice = [val[0] for val in xy]
xz_slice = [val[0] for val in xz]
yz_slice = [val[0] for val in yz]
xy_slice = xy_slice[len(xy_slice)//2]
xz_slice = xz_slice[len(xz_slice)//2]
yz_slice = yz_slice[len(yz_slice)//2]

im_xy = Image.fromarray(xy_slice.astype("uint8"))
im_xy = im_xy.convert("RGB")
im_xy.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/oskar_implemetation/ground_truth_slices_test/{args.subject}_xy_{args.tesla}_GT.jpg')
im_xz = Image.fromarray(xz_slice.astype("uint8"))
im_xz = im_xz.convert("RGB")
im_xz.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/oskar_implemetation/ground_truth_slices_test/{args.subject}_xz_{args.tesla}_GT.jpg')
im_yz = Image.fromarray(yz_slice.astype("uint8"))
im_yz = im_yz.convert("RGB")
im_yz.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/oskar_implemetation/ground_truth_slices_test/{args.subject}_yz_{args.tesla}_GT.jpg')

