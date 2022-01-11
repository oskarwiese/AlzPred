import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR_A = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/horse2zebra/train/A/"
TRAIN_DIR_B = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/horse2zebra/train/B/"
SAVE_IMG_DIR = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/saved_imgs"
SAVE_PLOTS_DIR = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/plots"
TRAIN_DIR = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/horse2zebra/train"  # ONLY FOR TRAINING ON HORSE2ZEBRA
FOLDER = "horse2zebra"
VAL_DIR = "data/val"
BATCH_SIZE = 1
DISC_LEARNING_RATE = 1e-5
GEN_LEARNING_RATE =  2e-4
LAMBDA_IDENTITY = 0
LAMBDA_CYCLE = 3
NUM_WORKERS = 4
NUM_EPOCHS = 800
IMG_SIZE = (256,256)
LOSS_SCALING = 1
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_A = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models/gen_a"
CHECKPOINT_GEN_B = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models/gen_b"
CHECKPOINT_DISC_A = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models/disc_a"
CHECKPOINT_DISC_B = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models/disc_b"
CHECKPOINT_LOSSES = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models/losses_run0.p"
CHECKPOINT_PARAMS = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models/config.json"
TRANSFORMS = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.CenterCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
GAUSS_STD = 0.15
STD_DECAY_RATE = 0.0015


################ Previous Models ################
#######      yz_nearest_lowlearning       #######
# BATCH_SIZE = 1
# DISC_LEARNING_RATE = 1e-7
# GEN_LEARNING_RATE = 1e-7
# LAMBDA_IDENTITY = 1.0
# LAMBDA_CYCLE = 10
# NUM_WORKERS = 4
# NUM_EPOCHS = 15
# IMG_SIZE = (256,256)
# LOSS_SCALING = 1

#######      xz_nearest_lowlearning       #######
# BATCH_SIZE = 1
# DISC_LEARNING_RATE = 1e-8
# GEN_LEARNING_RATE = 6e-8
# LAMBDA_IDENTITY = 1.0
# LAMBDA_CYCLE = 10
# NUM_WORKERS = 4
# NUM_EPOCHS = 15
# IMG_SIZE = (256,256)
# LOSS_SCALING = 1
#################################################