import torch
from PIL import Image
import imageio
import torchvision
from torchvision.datasets import folder
import config
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import nibabel as nib
from skimage import io
from nipy import load_image
from torchvision import transforms
import torchvision.datasets as datasets 
import torch.optim as optim
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_no_noise import Generator
from torch.utils.data import DataLoader
from PIL import Image
import warnings
import config
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=dev)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

if __name__ == '__main__':
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G_a = Generator(img_channels=1, num_residuals=9).to(dev)
    G_b = Generator(img_channels=1, num_residuals=9).to(dev)
    plane = 'yz'

    opti_G = optim.Adam(
        list(G_a.parameters()) + list(G_b.parameters()),
        lr = config.GEN_LEARNING_RATE,
        betas = (0.5, 0.999)
    )

    load_checkpoint(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/{plane}_nearest_lowlearning_gen_a.pth.tar', G_a, opti_G, config.GEN_LEARNING_RATE)
    load_checkpoint(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/{plane}_nearest_lowlearning_gen_b.pth.tar', G_b, opti_G, config.GEN_LEARNING_RATE)

    img_path = '/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/test_data_new/train/A/A/029_S_0866_xy_z_80_15T_test.png'
    im = Image.open(img_path) 
    data_transform=transforms.Compose([
                                transforms.Resize((256,256)),
                                transforms.CenterCrop((256,256)),
                                transforms.ToTensor(),
                           #     transforms.Grayscale(num_output_channels = 1),
                                transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize(0.5,0.5),
                                ])
    img = data_transform(im)
    img = img[None,:,:,:]
    
    # print(im.shape)
    # im = np.reshape(im, (1,1,256,256))
    # im = torch.tensor(im)
    # transforms.Normalize(0.5,0.5)
    # print(im.shape)    
    #im = torch.
    fake_img = G_b(img) # Fake 3T from 1.5 T
    print(fake_img.shape)
    save_image(fake_img*0.5 + 0.5, 'test.png')
    children = get_children(G_b)
    print(children)
