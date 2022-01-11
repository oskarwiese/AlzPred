import os
import sys
import datetime
import time
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from IPython import display
import matplotlib.pylab as plt
from pelutils import logger as log

sys.path.append("/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/oskar_implemetation")
from utils import save_checkpoint
import plots


def plot_loss(increment: list, loss1: list, loss2: list, path: str, label1: str, label2: str, title: str, ylabel: str):
    plt.figure(figsize=(15,4))
    plt.plot(increment, loss1, label = label1)
    plt.plot(increment, loss2, label = label2)
    plt.ylabel(ylabel)
    plt.xlabel('Steps')
    plt.legend(loc="upper right")
    plt.title(title)
    plt.savefig(f'{path}')
    plt.close()


if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 64
trainset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.genboi = nn.Sequential(
        
        nn.Linear(100, 2848),
        nn.BatchNorm1d(2848),
        nn.LeakyReLU(0.2),

        nn.Linear(2848, 2848),
        nn.BatchNorm1d(2848),
        nn.LeakyReLU(0.2),
            
        nn.Linear(2848, 2848),
        nn.BatchNorm1d(2848),
        nn.LeakyReLU(0.2),
            
        nn.Linear(2848, 2848),
        nn.BatchNorm1d(2848),
        nn.LeakyReLU(0.2),
            
        nn.Linear( 2848, 28**2),
        nn.Tanh()
            
                                )
            
        
    def forward(self, x):
        x = self.genboi(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dropout = 0.15
        self.disc = nn.Sequential(
            
            nn.Linear(28*28, 1024),
            nn.Dropout(p=self.dropout),
            nn.LeakyReLU(0.2),
            
            nn.Linear(1024,2048),
            nn.Dropout(p=self.dropout),
            nn.LeakyReLU(0.2),

            nn.Linear(2048,1024),
            nn.Dropout(p=self.dropout),
            nn.LeakyReLU(0.2),

            nn.Linear(1024,512),
            nn.Dropout(p=self.dropout),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 256),
            nn.Dropout(p=self.dropout),            
            nn.Linear(256, 1),
            
            
         
          #  nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.disc(x)
        return x





if __name__ == "__main__":
    #Initialize networks
    d = Discriminator().to(device)
    g = Generator().to(device)
    d_opt = torch.optim.Adam(d.parameters(), 0.0004, (0.5, 0.999))
    g_opt = torch.optim.Adam(g.parameters(), 0.00005, (0.5, 0.999))

    plt.figure(figsize=(20,10))
    subplots = [plt.subplot(2, 6, k+1) for k in range(12)]
    num_epochs = 200
    discriminator_final_layer = torch.sigmoid
    
    # Plotting variables
    increments = []
    g_losses = []
    d_losses = []
    increment = 0

    log.log.configure(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/log/{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_log.log')
    
    log.log.section("Start of training run")
    for epoch in range(num_epochs):
        start = time.time()
        for minibatch_no, (x, target) in enumerate(train_loader):
            iteration_time = time.time()
            x_real = x.to(device)*2-1 #scale to (-1, 1) range
            z = torch.randn(x.shape[0], 100).to(device)
            x_fake = g(z) 

            d_fake = d(x_fake)
            d_real = d(x_real)
            label_f = torch.zeros(d_fake.size(0)).unsqueeze(1).float().to(device)
            label_t = torch.ones(d_real.size(0)).unsqueeze(1).float().to(device)
            #Update discriminator
            
            d.zero_grad()
            
            loser = nn.MSELoss()
            # loser = nn.L1Loss()
            
            real_loss = loser(d(x_real), label_t)
            fake_loss = loser(d(x_fake), label_f )
            
            d_loss = real_loss + fake_loss
            d_loss.backward( retain_graph = True )
            d_opt.step()

            #Update generator
            g.zero_grad()
            loss_gen = nn.MSELoss()
            # loss_gen = nn.BCEWithLogitsLoss()
            
            g_loss = loss_gen(d(x_fake), label_t ) 
            g_loss.backward( retain_graph = True )
            g_opt.step()
            
            assert(not np.isnan(d_loss.item()))
            #Plot results for some minibatches
            plot_every = 500
            if minibatch_no % plot_every == 0:
                with torch.no_grad():
                    P = discriminator_final_layer(d(x_fake))
                    for k in range(11):
                        x_fake_k = x_fake[k].cpu().squeeze()/2+.5
                        subplots[k].imshow(x_fake_k, cmap='gray')
                        subplots[k].set_title('d(x)=%.2f' % P[k])
                        subplots[k].axis('off')
                    z = torch.randn(batch_size, 100).to(device)
                    H1 = discriminator_final_layer(d(g(z))).cpu()
                    H2 = discriminator_final_layer(d(x_real)).cpu()
                    plot_min = min(H1.min(), H2.min()).item()
                    plot_max = max(H1.max(), H2.max()).item()
                    subplots[-1].cla()
                    subplots[-1].hist(H1.squeeze(), color = ["red" for val in range(len(H1.squeeze()))], label='fake', range=(plot_min, plot_max), alpha=0.5)
                    subplots[-1].hist(H2.squeeze(), color = ["green" for val in range(len(H1.squeeze()))], label='real', range=(plot_min, plot_max), alpha=0.5)
                    subplots[-1].legend()
                    subplots[-1].set_xlabel('Probability of being real')
                    subplots[-1].set_title('Discriminator loss: %.2f' % d_loss.item())
                    
                    title = 'Epoch {e} - minibatch {n}/{d}'.format(e=epoch+1, n=minibatch_no, d=len(train_loader))
                    plt.gcf().suptitle(title, fontsize=20)
                    display.display(plt.gcf())
                    plt.savefig(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/plots/result_epoch_{epoch}_minibatch_{minibatch_no}.png')
                    display.clear_output(wait=True)
                
                log.log(f'Running batch {minibatch_no} took: {time.time() - iteration_time:.2f} seconds\n')
                log.log(f'G_loss epoch_{epoch} batch_{minibatch_no}: {g_loss}\n\n\n')
                log.log(f'D_loss epoch_{epoch} batch_{minibatch_no}: {d_loss}\n\n\n')

                d_losses.append(float(d_loss))
                g_losses.append(float(g_loss))
                increments.append(increment)
                plot_loss(increments, g_losses, d_losses, path = f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/plots/losses.png', label1 = "Generator Loss", label2 = "Discriminator Loss", title = "GAN General Losses", ylabel = "Loss")

                increment += 1 * plot_every

        log.log(f'Running epoch {epoch} took: {time.time() - start:.2f} seconds \n')
        
        if epoch % 20 == 0:
            save_checkpoint(g, g_opt, filename = f"/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/models/generator_epoch_{epoch}.pth.tar")
            save_checkpoint(d, d_opt, filename = f"/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/preliminary/models/discriminator_epoch_{epoch}.pth.tar")
        
                    