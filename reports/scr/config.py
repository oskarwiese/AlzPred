import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR_A = "/dtu-compute/ADNIbias/AlzPred/horse2zebra/train/A/"
TRAIN_DIR_B = "/dtu-compute/ADNIbias/AlzPred/horse2zebra/train/B/"
SAVE_IMG_DIR = "/dtu-compute/ADNIbias/AlzPred/git_code/AlzPred/oskar_implemetation"
VAL_DIR = "data/val"
BATCH_SIZE = 1
LOSS_PLOT = "/dtu-compute/ADNIbias/AlzPred/git_code/AlzPred/oskar_implemetation"
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
IMG_SIZE = (256,256)
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_GEN_A = "gen_a.pth.tar"
CHECKPOINT_GEN_B = "gen_b.pth.tar"
CHECKPOINT_DISC_A = "disc_a.pth.tar"
CHECKPOINT_DISC_B = "disc_b.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
