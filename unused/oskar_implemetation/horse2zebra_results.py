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

sys.path.append("/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/oskar_implemetation")
from utils import load_checkpoint, UnNormalize
from generator_no_noise import Generator
from discriminator_model import Discriminator
import plots
import config



if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1
train_a = datasets.ImageFolder(root=config.TRAIN_DIR_A,
                                transform=transforms.Compose([
                                    transforms.Resize(config.IMG_SIZE),
                                    transforms.CenterCrop(config.IMG_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]))
train_b = datasets.ImageFolder(root=config.TRAIN_DIR_B,
                                transform=transforms.Compose([
                                    transforms.Resize(config.IMG_SIZE),
                                    transforms.CenterCrop(config.IMG_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

dev = "cuda" if torch.cuda.is_available() else "cpu"

D_a = Discriminator(in_channels=3).to(dev)
D_b = Discriminator(in_channels=3).to(dev)
G_a = Generator(img_channels=3, num_residuals=9).to(dev)
G_b = Generator(img_channels=3, num_residuals=9).to(dev)
g_lr = config.GEN_LEARNING_RATE
d_lr = config.DISC_LEARNING_RATE

opti_g = optim.Adam(list(G_a.parameters()) + list(G_b.parameters()), lr = g_lr, betas = (0.5, 0.999))
opti_d = optim.Adam(list(D_a.parameters()) + list(D_b.parameters()), lr = d_lr, betas = (0.5, 0.999))

load_checkpoint('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models/gen_a_epoch_40_idx_0.pth.tar', G_a, opti_g, config.GEN_LEARNING_RATE)
load_checkpoint('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models/gen_b_epoch_40_idx_0.pth.tar', G_b, opti_g, config.GEN_LEARNING_RATE)
load_checkpoint('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models/disc_a_epoch_40_idx_0.pth.tar', D_a, opti_d, config.DISC_LEARNING_RATE)
load_checkpoint('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/models/disc_b_epoch_40_idx_0.pth.tar', D_b, opti_d, config.DISC_LEARNING_RATE)
with torch.no_grad():
    loader = zip(loader_a, loader_b)
    for idx, ((a,_), (b,_)) in enumerate(loader):
        a = a.to(config.DEVICE)
        fake_b = G_b(a)
        fake_a = G_a(fake_b)

        im_a = torch.squeeze(a)
        im_a = unorm(im_a)
        im_a = ToPILImage()(im_a)
        im_b = torch.squeeze(fake_b)
        im_b = unorm(im_b)
        im_b = ToPILImage()(im_b)
        im_a.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/output_imgs/real_a_{idx}.jpg')
        im_b.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/output_imgs/fake_b_{idx}.jpg')
        im_c = torch.squeeze(fake_a)
        im_c = unorm(im_c)
        im_c = ToPILImage()(im_c)
        im_c.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/output_imgs/fake_a_{idx}.jpg')

        if idx == 0:
            break



        # a = a.to(config.DEVICE)
        # b = b.to(config.DEVICE)
        # fake_a = G_a(b)

        # fake_b = G_b(a)

        # im_a = torch.squeeze(a)
        # im_a = unorm(im_a)
        # im_a = ToPILImage()(im_a)
        # im_b = torch.squeeze(b)
        # im_b = unorm(im_b)
        # im_b = ToPILImage()(im_b)
        # im_a.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/output_imgs/real_a_{idx}.jpg')
        # im_b.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/output_imgs/real_b_{idx}.jpg')

        # im_a = torch.squeeze(fake_a)
        # im_a = unorm(im_a)
        # im_a = ToPILImage()(im_a)
        # im_b = torch.squeeze(fake_b)
        # im_b = unorm(im_b)
        # im_b = ToPILImage()(im_b)
        # im_a.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/output_imgs/fake_a_{idx}.jpg')
        # im_b.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/horse2zebra_cyclegan/output_imgs/fake_b_{idx}.jpg')
    
        # Save real images
        # real_im = Image.fromarray((x[0][0].numpy() * 255).astype("uint8"))
        # real_im = real_im.convert("RGB")
        # real_im.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/output_imgs/MNIST_GAN_mse_real_{i}.jpg')
    
        # # Save discriminator prediction plot
        # z = torch.randn(batch_size, 100).to(device)
        # H1 = discriminator_final_layer(d(g(z))).cpu()
        # H2 = discriminator_final_layer(d(x_real)).cpu()
        # plot_min = min(H1.min(), H2.min()).item()
        # plot_max = max(H1.max(), H2.max()).item()
        # plt.hist(H1.squeeze(), color = ["red" for val in range(len(H1.squeeze()))], label='fake', range=(plot_min, plot_max), alpha=0.5)
        # plt.hist(H2.squeeze(), color = ["green" for val in range(len(H1.squeeze()))], label='real', range=(plot_min, plot_max), alpha=0.5)
        # plt.legend()
        # plt.xlabel('Probability of being real')
        # plt.title('Discriminator loss: %.2f' % d_loss.item())
        # plt.savefig(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/output_imgs/disc_preds_{i}.jpg')
        # plt.clf()
        # plt.cla()
        # plt.close()
    
        if idx >= 1:
            break