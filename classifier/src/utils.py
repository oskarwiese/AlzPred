import pandas as pd 
import torchvision 
import torchio
import torch 
import random
import numpy as np
from torch.utils.data import Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Test to see if GPU is available

class mydataLoader(Dataset):
    def __init__(self, paths, label, testvalortrain):
        self.testvalortrain = testvalortrain # Needs to be 0 or 1 
        self.image_paths = paths 
        self.labels = label
           
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index): #Notice model will become too big with 256x256x256 images, hence the resize
        transformR = torchvision.transforms.Compose([torchvision.transforms.RandomRotation(np.random.randint(0,15))])
        transformC = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop((256, 160)), torchvision.transforms.Resize((145,121))])
        transformD = torchvision.transforms.Compose([torchio.transforms.RandomElasticDeformation(np.random.randint(7,11), np.random.randint(11,16))])
        transformF = torchvision.transforms.Compose([torchio.transforms.RandomFlip(('P'), 1.0)])

        path = self.image_paths[index]
        label = self.labels[index]
        image = torchio.ScalarImage(path)
        image = np.array(image)
        image = torch.from_numpy(image).float()
        image = transformC(image)

        # Apply augmentation 0.8% of times
        to_aug_or_not_to_aug = ['yes', 'no']
        rand = random.choices(to_aug_or_not_to_aug, weights = [2, 8], k = 1)

        if ((rand[0] == 'yes') and (self.testvalortrain==0)):
            cAugment = np.random.randint(0,3)
            if (cAugment == 0):
                image = transformR(image)
            if (cAugment == 1):
                image = transformF(image)
            if (cAugment == 2):
                image = transformD(image)
        return image, label


def train_test_split(df, frac=0.2):
    
    # get random sample 
    test = df.sample(frac=frac, axis=0, random_state=42)

    # get everything but the test sample
    train = df.drop(index=test.index)

    return train, test

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr