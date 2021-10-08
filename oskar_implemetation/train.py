import torch
from dataset import HorseZebraDataset
import sys

from torchvision import transforms
import torchvision.datasets as datasets 
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_cykel(D_a, D_b, G_a, G_b, loader_a, loader_b, opti_D, opti_G, l1, mse, d_scaler, g_scaler):
    loader = zip(loader_a, loader_b)
    
    loop = tqdm(loader, leave = True)

    for idx, ((a,_), (b,_)) in enumerate(loop):
        a = a.to(config.DEVICE)
        b = b.to(config.DEVICE)

        # Train Discriminators

        with torch.cuda.amp.autocast():
            fake_b = G_b(a)
            D_b_real = D_b(b)
            D_b_fake = D_b(fake_b.detach())
            D_b_real_loss = mse(D_b_real, torch.ones_like(D_b_real))
            D_b_fake_loss = mse(D_b_fake, torch.zeros_like(D_b_fake))
            D_b_loss = D_b_real_loss + D_b_fake_loss

            fake_a = G_a(b)
            D_a_real = D_a(a)
            D_a_fake = D_a(fake_a.detach())
            D_a_real_loss = mse(D_a_real, torch.ones_like(D_a_real))
            D_a_fake_loss = mse(D_a_fake, torch.zeros_like(D_a_fake))
            D_a_loss = D_a_real_loss + D_a_fake_loss

            # To put it together

            D_loss = (D_b_loss + D_a_loss)/2 # In the paper they divide by 2

        opti_D.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opti_D)
        d_scaler.update()

        # Train Generators 
        
        with torch.cuda.amp.autocast():

            # Calculate adveserial loss for both generators

            D_b_fake = D_b(fake_a)
            D_a_fake = D_a(fake_b)

            loss_G_b = mse(D_b_fake, torch.ones_like(D_b_fake))
            loss_G_a = mse(D_a_fake, torch.ones_like(D_a_fake))

            # Cycle loss 
            cycle_a = G_a(fake_b)
            cycle_b = G_b(fake_a)
            cycle_a_loss = l1(a, cycle_a)
            cycle_b_loss = l1(b, cycle_b)

            # Identity loss
            identity_a = G_a(a)
            identity_b = G_b(b)
            identity_a_loss = l1(a, identity_a)
            identity_b_loss = l1(b, identity_b)

            # Add all together
            G_loss = (
                loss_G_a + loss_G_b + cycle_a_loss * config.LAMBDA_CYCLE + cycle_b_loss * config.LAMBDA_CYCLE + identity_a_loss * config.LAMBDA_IDENTITY + identity_b_loss * config.LAMBDA_IDENTITY
            )

        opti_G.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opti_G)
        g_scaler.update()

        if idx % 20 == 0:
            save_image(fake_b*0.5 + 0.5, config.SAVE_IMG_DIR + f'/saved_images/horse_idx_{idx}.png')
            save_image(fake_a*0.5 + 0.5, config.SAVE_IMG_DIR + f'/saved_images/zebra_idx_{idx}.png')






def main():
    D_a = Discriminator(in_channels=3).to(config.DEVICE)
    D_b = Discriminator(in_channels=3).to(config.DEVICE)
    G_a = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    G_b = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opti_D = optim.Adam(
        list(D_a.parameters()) + list(D_b.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5, 0.999) # Beta1 = 0.5 and Beta2 = 0.999 
    )

    opti_G = optim.Adam(
        list(G_a.parameters()) + list(G_b.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5, 0.999)
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_A, G_a, opti_G, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_B, G_b, opti_G, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_A, D_a, opti_D, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_B, D_b, opti_D, config.LEARNING_RATE)


    #dataset = HorseZebraDataset(
    #    root_horse = config.TRAIN_DIR + '/horses', root_zebra = config.TRAIN_DIR + '/zebras', transform = config.transforms
    #                            )

    train_a = datasets.ImageFolder(root=config.TRAIN_DIR_A,
                            transform=transforms.Compose([
                                transforms.Resize(config.IMG_SIZE),
                                transforms.CenterCrop(config.IMG_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    train_b = datasets.ImageFolder(root=config.TRAIN_DIR_B,
                            transform=transforms.Compose([
                                transforms.Resize(config.IMG_SIZE),
                                transforms.CenterCrop(config.IMG_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))


    loader_a = DataLoader(
        train_a,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    loader_b = DataLoader(
        train_b,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_cykel(D_a, D_b, G_a, G_b, loader_a, loader_b, opti_D, opti_G, l1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(G_a, opti_G, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(G_b, opti_G, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(D_a, opti_D, filename=config.CHECKPOINT_DISC_A)
            save_checkpoint(D_b, opti_D, filename=config.CHECKPOINT_DISC_B)

if __name__ == '__main__':
    main()
