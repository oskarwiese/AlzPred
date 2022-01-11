import torch
import datetime
#from dataset import HorseZebraDataset
import sys
import time
from torchvision import transforms
import torchvision.datasets as datasets 
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import loader_config
import plots
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
global increments
global aggregated_loss
from pelutils import logger as log
import pickle

def load():
    start = time.time()


    log.log('Starting to create dataset')
    train_a = datasets.ImageFolder(root=loader_config.TRAIN_DIR_A,
                            transform=transforms.Compose([
                                transforms.Resize(loader_config.IMG_SIZE),
                                transforms.CenterCrop(loader_config.IMG_SIZE),
                                transforms.ToTensor(),
                                transforms.Grayscale(num_output_channels = 1),
                                transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize(0.5,0.5),
                                ]))

    train_b = datasets.ImageFolder(root=loader_config.TRAIN_DIR_B,
                            transform=transforms.Compose([
                                transforms.Resize(loader_config.IMG_SIZE),
                                transforms.CenterCrop(loader_config.IMG_SIZE),
                                transforms.ToTensor(),
                                transforms.Grayscale(num_output_channels = 1),
                                transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize(0.5,0.5),
                                ]))


    loader_a = DataLoader(
        train_a,
        batch_size=1,
        shuffle=True,
        num_workers=10,
        pin_memory=True
    )

    loader_b = DataLoader(
        train_b,
        batch_size=1,
        shuffle=True,
        num_workers=10,
        pin_memory=True
    )
    torch.save(loader_a, f"/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/no_black/{loader_config.FOLDER}/loader_a_torch.pt")
    torch.save(loader_b, f"/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/no_black/{loader_config.FOLDER}/loader_b_torch.pt")
#    pickle.dump(loader_a, open(f"/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/no_black/{loader_config.FOLDER}/loader_a.p", "wb"))  # save it into a file
#    pickle.dump(loader_b, open(f"/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/no_black/{loader_config.FOLDER}/loader_b.p", "wb"))  # save it into a file 

    finished = time.time() - start
    log.log(f'Creating dataset took: {finished:.2f} seconds\n')


if __name__ == '__main__':
    log.log.configure(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/log/{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_{loader_config.FOLDER}_data_loader.log')
    log.log('Hello Pussy!')
    load() # Load mi boi 
