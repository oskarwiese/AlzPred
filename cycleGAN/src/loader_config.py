import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR_A = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/no_black/yz/train/A/"
TRAIN_DIR_B = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/no_black/yz/train/B/"
IMG_SIZE = (256,256)
FOLDER = "yz"
