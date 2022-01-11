import torch
import datetime
import csv
import sys
import time
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.datasets as datasets 
from utils import save_checkpoint, load_checkpoint, average_nth, UnNormalize
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
import plots
import pickle
from tqdm import tqdm as tqdm
from torchvision.utils import save_image
from discriminator_model_horse2zebra import Discriminator, GaussianNoise
from pelutils import logger as log
import json
import argparse
from dataset import HorseZebraDataset
from generator_horse2zebra import Generator

def decay_gauss_std(net):
    std = 0
    for m in net.modules():
        if isinstance(m, GaussianNoise):
            m.decay_step()
            std = m.std
    log.log({f'std set to: {std}'})

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch, increment):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)
    iteration_time = time.time()

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # add all togethor
            G_loss = (
                3*loss_G_Z
                + 3*loss_G_H
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
                + identity_horse_loss * config.LAMBDA_IDENTITY
                + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        log_every = 667 # Should be 20000 when running code 
        plot_every = 10
        save_every = 10000
        save_every_epoch = 20
        average_every = 40

        if idx % log_every == 0:
            log.log(f'loss_G_a: {loss_G_H} \nloss_G_b: {loss_G_Z} \nloss_D_a: {D_H_loss} \nloss_D_b: {D_Z_loss} \ncycle_loss_a: {cycle_horse_loss} \ncycle_loss_b: {cycle_zebra_loss} \nidentity_loss_a: {identity_horse_loss} \nidentity_loss_b: {identity_zebra_loss}')
            log.log(f'G_loss epoch_{epoch} idx_{idx}: {G_loss:.2f}\n\n\n')
            with torch.no_grad():
                save_image(fake_horse*0.5+0.5, config.SAVE_IMG_DIR + f'/horse/{epoch}_{idx}_fake.png')
                save_image(fake_zebra*0.5+0.5, config.SAVE_IMG_DIR + f'/zebra/{epoch}_{idx}_fake.png')
                save_image(zebra*0.5 + 0.5, config.SAVE_IMG_DIR + f'/horse/{epoch}_{idx}_real.png')
                save_image(horse*0.5 + 0.5, config.SAVE_IMG_DIR + f'/zebra/{epoch}_{idx}_real.png')

        if idx % plot_every == 0:
            increments.append(increment)
            G_losses.append(float(G_loss / config.LOSS_SCALING))
            D_losses.append(float(D_loss / config.LOSS_SCALING))
            G_b_losses.append(float(loss_G_Z))
            G_a_losses.append(float(loss_G_H))
            D_b_losses.append(float(D_Z_loss))
            D_a_losses.append(float(D_H_loss))
            cycle_a_losses.append(float(cycle_horse_loss)) 
            cycle_b_losses.append(float(cycle_zebra_loss))
            identity_losses_a.append(float(identity_horse_loss))
            identity_losses_b.append(float(identity_zebra_loss))

                
            if idx != 0:
                # increment: list, loss1: list, loss2: list, filename: str, label1: str, label2: str, title: str, ylabel: str
                plots.plot_loss(average_nth(increments, average_every), average_nth(G_losses, average_every), average_nth(D_losses, average_every), filename = config.FOLDER + f'_general_losses.png', label1 = "Total Generator Loss", label2 = "Total Discriminator Loss", title = "CycleGAN General Losses", ylabel = "Loss")
                plots.plot_loss(average_nth(increments, average_every), average_nth(G_a_losses, average_every), average_nth(G_b_losses, average_every), filename = config.FOLDER + f'_generator_losses.png', label1 = r"$G_A$ " + "Loss", label2 = r"$G_B$ " + "Loss", title = "CycleGAN Generator Losses", ylabel = "Loss")
                plots.plot_loss(average_nth(increments, average_every), average_nth(D_a_losses, average_every), average_nth(D_b_losses, average_every), filename = config.FOLDER + f'_discriminator_losses.png', label1 = r"$D_A$ " + "Loss", label2 = r"$D_B$ " + "Loss", title = "CycleGAN  Losses", ylabel = "Loss")
                plots.plot_loss(average_nth(increments, average_every), average_nth(cycle_a_losses, average_every), average_nth(cycle_b_losses, average_every), filename = config.FOLDER + f'_cycle_losses.png', label1 = r"$Cyc_A$ " + "Loss", label2 = r"$Cyc_B$ " + "Loss", title = "CycleGAN Cycle-consistency Losses", ylabel = "Loss")
                plots.plot_loss(average_nth(increments, average_every), average_nth(identity_losses_a, average_every), average_nth(identity_losses_b, average_every), filename = config.FOLDER + f'_identity_losses.png', label1 = r"$Iden_A$ " + "Loss", label2 = r"$Iden_B$ " + "Loss", title = "CycleGAN Identity Losses", ylabel = "Loss")
                plt.clf()
                plt.cla()
                plt.close()

            # log.log(f'Running iteration {increment} took: {time.time() - iteration_time:.2f} seconds\n')
            increment += 1 * plot_every
            # iteration_time = time.time()

        if config.SAVE_MODEL:
            if epoch % save_every_epoch == 0 and idx == 0:
                save_checkpoint(gen_H, opt_gen, filename=f"{config.CHECKPOINT_GEN_A}_{epoch}_{idx}.pth.tar")
                save_checkpoint(gen_Z, opt_gen, filename=f"{config.CHECKPOINT_GEN_B}_{epoch}_{idx}.pth.tar")
                save_checkpoint(disc_H, opt_disc, filename=f"{config.CHECKPOINT_DISC_A}_{epoch}_{idx}.pth.tar")
                save_checkpoint(disc_Z, opt_disc, filename=f"{config.CHECKPOINT_DISC_B}_{epoch}_{idx}.pth.tar")

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

        loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))
    
    decay_gauss_std(disc_H)
    decay_gauss_std(disc_Z)
    return increment



def main():
    log.log.configure(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/log/{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_log.log')
    #Change in channels to 1 for grey scale images
    log.log.section('Device')
    log.log(f'Device is: {config.DEVICE}')

    disc_H = Discriminator(in_channels=3, std=config.GAUSS_STD, std_decay_rate=config.STD_DECAY_RATE).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3, std=config.GAUSS_STD, std_decay_rate=config.STD_DECAY_RATE).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

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
        'GAUSS_STD': config.GAUSS_STD,
        'STD_DECAY_RATE': config.STD_DECAY_RATE
    } # For model configs
    
    if config.SAVE_MODEL:
        with open(config.CHECKPOINT_PARAMS,'w') as f:
            json.dump(config_params, f, indent = 4)

    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.DISC_LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.GEN_LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.GEN_LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.GEN_LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.DISC_LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.DISC_LEARNING_RATE,
        )

    start = time.time()
    log.log('Starting to create dataset')
    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR+"/horses", root_zebra=config.TRAIN_DIR+"/zebras", transform=config.TRANSFORMS)
    val_dataset = HorseZebraDataset(
       root_horse="/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/horse2zebra/test/horses", root_zebra="/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/horse2zebra/test/zebras", transform=config.TRANSFORMS)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    finished = time.time() - start
    log.log(f'Creating dataset took: {finished:.2f} seconds\n')

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    inc = 0
    log.log.section("Start of training run")
    for epoch in range(config.NUM_EPOCHS):
        start = time.time()
        inc = train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch, inc)
        log.log(f'Running epoch {epoch} took: {time.time() - start:.2f} seconds \n')

        # if config.SAVE_MODEL:
        #     save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
        #     save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
        #     save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
        #     save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":    
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