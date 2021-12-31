import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch.optim as optim
from train_GAN import Generator, Discriminator
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from PIL import Image

sys.path.append("/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/oskar_implemetation")
from utils import load_checkpoint
import plots


if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 20
testset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

g = Generator().to(device)
d = Discriminator().to(device)
g_lr = 0.00005
d_lr = 0.0004

opti_g = optim.Adam(
    list(g.parameters()),
    lr = g_lr,
    betas = (0.5, 0.999)
)
opti_d = optim.Adam(
    list(d.parameters()),
    lr = d_lr,
    betas = (0.5, 0.999)
)

discriminator_final_layer = torch.sigmoid

load_checkpoint(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/models_normalloss/generator_epoch_199.pth.tar', g, opti_g, g_lr)
load_checkpoint(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/models_normalloss/discriminator_epoch_199.pth.tar', d, opti_d, d_lr)
with torch.no_grad():
    for i, (x, target) in enumerate(test_loader):
        z = torch.randn(batch_size, 100).to(device)
        x_real = x.to(device)*2-1 #scale to (-1, 1) range
        x_fake = g(z) 
        x_fake_k = x_fake[0].cpu().squeeze()/2+.5
        d_fake = d(x_fake)
        d_real = d(x_real)
        label_f = torch.zeros(d_fake.size(0)).unsqueeze(1).float().to(device)
        label_t = torch.ones(d_real.size(0)).unsqueeze(1).float().to(device)
        
        loser = nn.MSELoss()
        
        real_loss = loser(d(x_real), label_t)
        fake_loss = loser(d(x_fake), label_f )
        
        d_loss = real_loss + fake_loss


        # Save fake images
        im = Image.fromarray((x_fake_k.numpy() * 255).astype("uint8"))
        im = im.convert("RGB")
        im.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/output_imgs/MNIST_GAN_mse_fake_{i}.jpg')
        
        # Save real images
        real_im = Image.fromarray((x[0][0].numpy() * 255).astype("uint8"))
        real_im = real_im.convert("RGB")
        real_im.save(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/output_imgs/MNIST_GAN_mse_real_{i}.jpg')
        
        # Save discriminator prediction plot
        z = torch.randn(batch_size, 100).to(device)
        H1 = discriminator_final_layer(d(g(z))).cpu()
        H2 = discriminator_final_layer(d(x_real)).cpu()
        plot_min = min(H1.min(), H2.min()).item()
        plot_max = max(H1.max(), H2.max()).item()
        plt.hist(H1.squeeze(), color = ["red" for val in range(len(H1.squeeze()))], label='fake', range=(plot_min, plot_max), alpha=0.5)
        plt.hist(H2.squeeze(), color = ["green" for val in range(len(H1.squeeze()))], label='real', range=(plot_min, plot_max), alpha=0.5)
        plt.legend()
        plt.xlabel('Probability of being real')
        plt.title('Discriminator loss: %.2f' % d_loss.item())
        plt.savefig(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/output_imgs/disc_preds_{i}.jpg')
        plt.clf()
        plt.cla()
        plt.close()
        
        if i >= 7:
            break