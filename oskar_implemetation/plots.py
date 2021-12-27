import numpy as np
import matplotlib.pyplot as plt 

def plot_loss(G_loss: list, D_loss: list, increment: list):
    plt.plot(increment, G_loss, label = "Generator loss")
    plt.plot(increment, D_loss, label = "Discriminator loss")
    plt.ylabel('Training Loss')
    plt.xlabel('Increment')
    plt.legend(loc="upper right")
    plt.title(r'$\mathrm{cycleGAN \ Model \ Loss }$')
    #plt.savefig('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/oskar_implemetation/plots/training_loss.png')

if __name__ == "__main__":
    xs = np.linspace(1, 2, 100)
    ys1 = [np.exp(x) for x in xs]
    ys2 = [np.log(x) for x in xs]
    plot_loss(xs, ys1, ys2)
    plt.show()