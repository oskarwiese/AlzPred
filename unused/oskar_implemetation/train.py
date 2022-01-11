import torch
import datetime
import csv
import sys
import time
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.datasets as datasets 
from utils import save_checkpoint, load_checkpoint, average_nth
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
import plots
import pickle
from tqdm import tqdm as tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_no_noise import Generator
from pelutils import logger as log
import json
import argparse

def train_cykel(D_a, D_b, G_a, G_b, loader_a, loader_b, opti_D, opti_G, l1, mse, d_scaler, g_scaler, epoch, increment):
    loader = zip(loader_a, loader_b)
    iteration_time = time.time()
    
    for idx, ((a,_), (b,_)) in enumerate(loader):
        a = a.to(config.DEVICE)
        b = b.to(config.DEVICE)
        #log.log(f'Shape of a: {a.shape}\nShape of b: {b.shape}\n')
        #log.log(f'Sum of a: {a.sum()}\nSum of b: {b.sum()}\n')
        #log.log(f'Min, Max of a is: {a.min(), a.max()}\nMin, Max of b is: {b.min(), b.max()}\n\n')
        
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
            D_loss = config.LOSS_SCALING * (D_b_loss + D_a_loss)/2  # In the paper they divide by 2

            # Calculate accuracy
            soft = torch.nn.Softmax(1)
            D_a_acc = (
                torch.sum(soft(D_a_real) == torch.ones_like(D_a_real)) + 
                torch.sum(soft(D_a_fake) == torch.zeros_like(D_a_fake))) / (2*30**2)

            D_b_acc = (
                torch.sum(soft(D_b_real) == torch.ones_like(D_b_real)) + 
                torch.sum(soft(D_b_fake) == torch.zeros_like(D_b_fake))) / (2*30**2)

        if D_b_acc <= 0.5 and D_a_acc <= 0.5:
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
            G_loss = config.LOSS_SCALING * (
                loss_G_a + loss_G_b + cycle_a_loss * config.LAMBDA_CYCLE + cycle_b_loss * config.LAMBDA_CYCLE + identity_a_loss * config.LAMBDA_IDENTITY + identity_b_loss * config.LAMBDA_IDENTITY
            )

        opti_G.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opti_G)
        g_scaler.update()
        log_every = 1000 # Should be 20000 when running code 
        plot_every = 10
        save_every = 5000
        average_every = 40

        if idx % log_every == 0:
                log.log(f'loss_G_b: {loss_G_b} \nloss_G_a: {loss_G_a} \nloss_D_a: {D_a_loss} \nloss_D_b: {D_b_loss} \ncycle_loss_a: {cycle_a_loss} \ncycle_loss_b: {cycle_b_loss} \n')
                log.log(f'G_loss epoch_{epoch} idx_{idx}: {G_loss:.2f}\n\n\n')
                save_image(fake_b*0.5 + 0.5, config.SAVE_IMG_DIR + f'/1.5/1.5T_epoch_{epoch}_idx_{idx}.png')
                save_image(fake_a*0.5 + 0.5, config.SAVE_IMG_DIR + f'/3/3T_epoch_{epoch}_idx_{idx}.png')

        if idx % plot_every == 0:
                increments.append(increment)
                G_losses.append(float(G_loss / config.LOSS_SCALING))
                D_losses.append(float(D_loss / config.LOSS_SCALING))
                G_b_losses.append(float(loss_G_b))
                G_a_losses.append(float(loss_G_a))
                D_b_losses.append(float(D_b_loss))
                D_a_losses.append(float(D_a_loss))
                cycle_a_losses.append(float(cycle_a_loss)) 
                cycle_b_losses.append(float(cycle_b_loss))
                identity_losses_a.append(float(identity_a_loss))
                identity_losses_b.append(float(identity_b_loss))
                D_a_accs.append(float(D_a_acc))
                D_b_accs.append(float(D_b_acc))

                
                if idx != 0:
                    # increment: list, loss1: list, loss2: list, filename: str, label1: str, label2: str, title: str, ylabel: str
                    plots.plot_loss(average_nth(increments, average_every), average_nth(G_losses, average_every), average_nth(D_losses, average_every), filename = config.FOLDER + f'_general_losses.png', label1 = "Total Generator Loss", label2 = "Total Discriminator Loss", title = "CycleGAN General Losses", ylabel = "Loss")
                    plots.plot_loss(average_nth(increments, average_every), average_nth(G_a_losses, average_every), average_nth(G_b_losses, average_every), filename = config.FOLDER + f'_generator_losses.png', label1 = r"$G_A$ " + "Loss", label2 = r"$G_B$ " + "Loss", title = "CycleGAN Generator Losses", ylabel = "Loss")
                    plots.plot_loss(average_nth(increments, average_every), average_nth(D_a_losses, average_every), average_nth(D_b_losses, average_every), filename = config.FOLDER + f'_discriminator_losses.png', label1 = r"$D_A$ " + "Loss", label2 = r"$D_B$ " + "Loss", title = "CycleGAN  Losses", ylabel = "Loss")
                    plots.plot_loss(average_nth(increments, average_every), average_nth(cycle_a_losses, average_every), average_nth(cycle_b_losses, average_every), filename = config.FOLDER + f'_cycle_losses.png', label1 = r"$Cyc_A$ " + "Loss", label2 = r"$Cyc_B$ " + "Loss", title = "CycleGAN Cycle-consistency Losses", ylabel = "Loss")
                    plots.plot_loss(average_nth(increments, average_every), average_nth(identity_losses_a, average_every), average_nth(identity_losses_b, average_every), filename = config.FOLDER + f'_identity_losses.png', label1 = r"$Iden_A$ " + "Loss", label2 = r"$Iden_B$ " + "Loss", title = "CycleGAN Identity Losses", ylabel = "Loss")
                    plots.plot_loss(average_nth(increments, average_every), average_nth(D_a_accs, average_every), average_nth(D_b_accs, average_every), filename = config.FOLDER + f'_discriminator_accs.png', label1 = r"$D_a$ " + "Accuracy", label2 = r"$D_b$ " + "Accuracy", title = "CycleGAN Discriminator Accuracies", ylabel = "Accuracy")
                    plt.clf()
                    plt.cla()
                    plt.close()

                log.log(f'Running iteration {increment} took: {time.time() - iteration_time:.2f} seconds\n')
                increment += 1 * plot_every
                iteration_time = time.time()

        if config.SAVE_MODEL:
            if idx % save_every == 0:
                save_checkpoint(G_a, opti_G, filename=config.CHECKPOINT_GEN_A)
                save_checkpoint(G_b, opti_G, filename=config.CHECKPOINT_GEN_B)
                save_checkpoint(D_a, opti_D, filename=config.CHECKPOINT_DISC_A)
                save_checkpoint(D_b, opti_D, filename=config.CHECKPOINT_DISC_B)

                losses_dic = {}

                names = [
                    'increments',
                    'G_losses'
                    'D_losses'
                    'G_b_losses'
                    'G_a_losses'
                    'D_b_losses'
                    'D_a_losses'
                    'cycle_a_losses' 
                    'cycle_b_losses'
                    'identity_losses_a'
                    'identity_losses_b'
                    'D_a_accs'
                    'D_b_accs'
                     ]

                lists = [
                       increments, 
                        G_losses, 
                        D_losses, 
                        G_b_losses, 
                        G_a_losses, 
                        D_b_losses, 
                        D_a_losses, 
                        cycle_a_losses, 
                        cycle_b_losses, 
                        identity_losses_a, 
                        identity_losses_b, 
                        D_a_accs, 
                        D_b_accs
                ]
                for name, lsts in zip(names, lists):
                    losses_dic[name] = lsts

                pickle.dump(losses_dic, open(config.CHECKPOINT_LOSSES , "wb"))     
                

    return increment

def main():
    log.log.configure(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/log/{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_log.log')
    #Change in channels to 1 for grey scale images
    log.log.section('Device')
    log.log(f'Device is: {config.DEVICE}')
    D_a = Discriminator(in_channels=1).to(config.DEVICE)
    D_b = Discriminator(in_channels=1).to(config.DEVICE)
    G_a = Generator(img_channels=1, num_residuals=9).to(config.DEVICE)  # Konverterer fra 3T->1.5T
    G_b = Generator(img_channels=1, num_residuals=9).to(config.DEVICE)  # Konverterer fra 1.5T->3T

    parser = argparse.ArgumentParser() ##
    parser.add_argument("-d", '--dataset', type=str, help="horse: Horse2Zebra dataset. Leave empty for MRI images")
    args = parser.parse_args()

    if args.dataset == "horse":
        print("Training on horse dataset.")
    else:
        args.dataset = "MRI"
        print("Training on MRI dataset.")

    config_params = {
        'TRAIN_DIR_A' :config.TRAIN_DIR_A, 
        'TRAIN_DIR_B' :config.TRAIN_DIR_B, 
        'SAVE_IMG_DIR' :config.SAVE_IMG_DIR, 
        'FOLDER' :config.FOLDER, 
        'VAL_DIR' :config.VAL_DIR, 
        'BATCH_SIZE' :config.BATCH_SIZE, 
        'DISC_LEARNING_RATE' :config.DISC_LEARNING_RATE, 
        'GEN_LEARNING_RATE' :config.GEN_LEARNING_RATE, 
        'LAMBDA_IDENTITY' :config.LAMBDA_IDENTITY, 
        'LAMBDA_CYCLE' :config.LAMBDA_CYCLE, 
        'NUM_WORKERS' :config.NUM_WORKERS, 
        'NUM_EPOCHS' :config.NUM_EPOCHS, 
        'IMG_SIZE' :config.IMG_SIZE, 
        'LOSS_SCALING' :config.LOSS_SCALING, 
        'LOAD_MODEL' :config.LOAD_MODEL, 
        'SAVE_MODEL' :config.SAVE_MODEL, 
        'CHECKPOINT_GEN_A' :config.CHECKPOINT_GEN_A, 
        'CHECKPOINT_GEN_B' :config.CHECKPOINT_GEN_B, 
        'CHECKPOINT_DISC_A' :config.CHECKPOINT_DISC_A,
        'CHECKPOINT_DISC_B' :config.CHECKPOINT_DISC_B,
        'CHECKPOINT_LOSSES' :config.CHECKPOINT_LOSSES,
    } # For model configs
    
    if config.SAVE_MODEL:
        with open(config.CHECKPOINT_PARAMS,'w') as f:
            json.dump(config_params, f, indent = 4)

    opti_D = optim.Adam(
        list(D_a.parameters()) + list(D_b.parameters()),
        lr = config.DISC_LEARNING_RATE,
        betas = (0.5, 0.999) # Beta1 = 0.5 and Beta2 = 0.999 
    )

    opti_G = optim.Adam(
        list(G_a.parameters()) + list(G_b.parameters()),
        lr = config.GEN_LEARNING_RATE,
        betas = (0.5, 0.999)
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_A, G_a, opti_G, config.GEN_LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_B, G_b, opti_G, config.GEN_LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_A, D_a, opti_D, config.DISC_LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_B, D_b, opti_D, config.DISC_LEARNING_RATE)

    if args.dataset == "horse":
        dataset = HorseZebraDataset(
        root_horse = config.TRAIN_DIR + '/horses', root_zebra = config.TRAIN_DIR + '/zebras', transform = config.transforms
                                    )
        start = time.time()
        log.log('Starting to create dataset')
        train_a = datasets.ImageFolder(root=config.TRAIN_DIR_A,
                                transform=transforms.Compose([
                                    transforms.Resize(config.IMG_SIZE),
                                    transforms.CenterCrop(config.IMG_SIZE),
                                    transforms.ToTensor(),
                                    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.Grayscale(num_output_channels = 1),
                                    transforms.ConvertImageDtype(torch.float),
                                    transforms.Normalize(0.5,0.5),
                                    ]))
        train_b = datasets.ImageFolder(root=config.TRAIN_DIR_B,
                                transform=transforms.Compose([
                                    transforms.Resize(config.IMG_SIZE),
                                    transforms.CenterCrop(config.IMG_SIZE),
                                    transforms.ToTensor(),
                                    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.Grayscale(num_output_channels = 1),
                                    transforms.ConvertImageDtype(torch.float),
                                    transforms.Normalize(0.5,0.5),
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
        finished = time.time() - start
        log.log(f'Creating dataset took: {finished:.2f} seconds\n')

    else:
        loader_a = torch.load(f"/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/no_black/{config.FOLDER}/loader_a_torch.pt")
        loader_b = torch.load(f"/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/data/no_black/{config.FOLDER}/loader_b_torch.pt")
        log.log(f'Number of 1.5T images: {len(loader_a)}\n')
        log.log(f'Number of 3T images: {len(loader_b)}\n')


    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    inc = 0
    log.log.section("Start of training run")
    for epoch in range(config.NUM_EPOCHS):
        start = time.time()
        inc = train_cykel(D_a, D_b, G_a, G_b, loader_a, loader_b, opti_D, opti_G, l1, mse, d_scaler, g_scaler, epoch, inc)
        log.log(f'Running epoch {epoch} took: {time.time() - start:.2f} seconds \n')

        if config.SAVE_MODEL:
            save_checkpoint(G_a, opti_G, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(G_b, opti_G, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(D_a, opti_D, filename=config.CHECKPOINT_DISC_A)
            save_checkpoint(D_b, opti_D, filename=config.CHECKPOINT_DISC_B)

if __name__ == '__main__':
    increments = []
    G_losses = []
    D_losses = []
    G_b_losses = []
    G_a_losses = []
    D_b_losses = []
    D_a_losses = []
    cycle_a_losses = []
    cycle_b_losses = []
    identity_losses_a = []
    identity_losses_b = []
    D_a_accs = []
    D_b_accs = []
    main()
