import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from PIL import Image
from torchvision.transforms import ToPILImage
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse

sys.path.append("/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/oskar_implemetation")
sys.path.append("/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models_gaussnoise")
from utils import load_checkpoint, UnNormalize
from dataset import HorseZebraDataset
import plots
# import config

def plot_result(var, path):
    im = torch.squeeze(var)
    im = unorm(im)
    im = ToPILImage()(im)
    im.save(path)

parser = argparse.ArgumentParser() ##
parser.add_argument("-t", '--type', type=str,
                help="Type of results to generate: pretrained or otherwise")
args = parser.parse_args()

if not args.type:
    sys.exit('You should set the model loaded to be "pretrained" or "trained".')

if args.type != "pretrained":
    print("The images will be generated from the trained model.")
    args.type = "trained"
else:
    print("The images will be generated from the pretrained model.")


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

print("Reading json file.")
json_file_path = "/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models_gaussnoise/config.json"
with open(json_file_path, 'r') as j:
     config = json.load(j)

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1

print("Loading dataset.")
# dataset = HorseZebraDataset(
#         root_horse="/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/horse2zebra/train/horses", root_zebra="/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/horse2zebra/train/zebras", transform=TRANSFORMS)
# val_loader = DataLoader(
#         dataset,
#         batch_size=config["BATCH_SIZE"],
#         shuffle=True,
#         num_workers=config["NUM_WORKERS"],
#         pin_memory=True
#     )

val_dataset = HorseZebraDataset(
       root_horse="/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/horse2zebra/test/horses1", root_zebra="/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/horse2zebra/test/zebras1", transform=TRANSFORMS)
val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

dev = "cuda" if torch.cuda.is_available() else "cpu"

if args.type == "pretrained":
    from generator_aladdin import Generator
    from discriminator_model import Discriminator
    D_a = Discriminator(in_channels=3).to(dev)
    D_b = Discriminator(in_channels=3).to(dev)
    G_a = Generator(img_channels=3, num_residuals=9).to(dev)
    G_b = Generator(img_channels=3, num_residuals=9).to(dev)
    g_lr = config["GEN_LEARNING_RATE"]
    d_lr = config["DISC_LEARNING_RATE"]
else:
    from generator_horse2zebra import Generator
    from discriminator_model_horse2zebra import Discriminator
    D_a = Discriminator(in_channels=3, std=config["GAUSS_STD"], std_decay_rate=config["STD_DECAY_RATE"]).to(dev)
    D_b = Discriminator(in_channels=3, std=config["GAUSS_STD"], std_decay_rate=config["STD_DECAY_RATE"]).to(dev)
    G_a = Generator(img_channels=3, num_residuals=9).to(dev)
    G_b = Generator(img_channels=3, num_residuals=9).to(dev)
    g_lr = config["GEN_LEARNING_RATE"]
    d_lr = config["DISC_LEARNING_RATE"]

opti_g = optim.Adam(list(G_a.parameters()) + list(G_b.parameters()), lr = g_lr, betas = (0.5, 0.999))
opti_d = optim.Adam(list(D_a.parameters()) + list(D_b.parameters()), lr = d_lr, betas = (0.5, 0.999))


if args.type == "pretrained":
    load_checkpoint('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models_pretrained/genh.pth.tar', G_a, opti_g, 1e-5)
    load_checkpoint('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models_pretrained/genz.pth.tar', G_b, opti_g, 1e-5)
    load_checkpoint('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models_pretrained/critich.pth.tar', D_a, opti_d, 1e-5)
    load_checkpoint('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models_pretrained/criticz.pth.tar', D_b, opti_d, 1e-5)
else:
    model = "gaussnoise"
    epoch = "140"
    load_checkpoint(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models_{model}/gen_a_{epoch}_0.pth.tar', G_a, opti_g, config["GEN_LEARNING_RATE"])
    load_checkpoint(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models_{model}/gen_b_{epoch}_0.pth.tar', G_b, opti_g, config["GEN_LEARNING_RATE"])
    load_checkpoint(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models_{model}/disc_a_{epoch}_0.pth.tar', D_a, opti_d, config["DISC_LEARNING_RATE"])
    load_checkpoint(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models_{model}/disc_b_{epoch}_0.pth.tar', D_b, opti_d, config["DISC_LEARNING_RATE"])
    

with torch.no_grad():
    for idx, (b, a) in enumerate(val_loader):
        if idx >= 60:
            break
            
        print(f"Generating sample {idx}.")
        # Testing zebra generation and a->b->a cycle
        a = a.to(dev)
        fake_b = G_b(a)
        fake_a = G_a(fake_b)
        plot_result(a, f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/output_imgs/{model}/a-b/{idx}_{epoch}_real_a.jpg')
        plot_result(fake_b, f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/output_imgs/{model}/a-b/{idx}_{epoch}_fake_b.jpg')
        plot_result(fake_a, f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/output_imgs/{model}/a-b/{idx}_{epoch}_fake_a.jpg')
        
        # Testing horse generation and b->a->b cycle
        b = b.to(dev)
        fake_a = G_a(b)
        fake_b = G_b(fake_a)
        plot_result(b, f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/output_imgs/{model}/b-a/{idx}_{epoch}_real_a.jpg')
        plot_result(fake_a, f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/output_imgs/{model}/b-a/{idx}_{epoch}_fake_a.jpg')
        plot_result(fake_b, f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/output_imgs/{model}/b-a/{idx}_{epoch}_fake_b.jpg')