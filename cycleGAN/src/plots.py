import numpy as np
import matplotlib.pyplot as plt 
import config

def plot_loss(increment: list, loss1: list, loss2: list, filename: str, label1: str, label2: str, title: str, ylabel: str):
    plt.figure(figsize=(15,4))
    plt.plot(increment, loss1, label = label1)
    plt.plot(increment, loss2, label = label2)
    plt.ylabel(ylabel)
    plt.xlabel('Steps')
    plt.legend(loc="upper right")
    plt.title(title)
    plt.savefig(f'{config.SAVE_PLOTS_DIR}/{filename}')
    plt.clf()
    plt.cla()
    plt.close()

if __name__ == "__main__":
    xs = np.linspace(1, 2, 100)
    ys1 = [np.expm1(x) for x in xs]
    ys2 = [np.tanh(x) for x in xs]
    plot_loss_general(increment = xs, loss1 = ys1, loss2 = ys2, filename = '_D_b_acc.png', label1 = "Exponential Function", label2 = "Hyperbolic Tangent", title = "Showing how this works", ylabel = "units or something")
    plt.show()