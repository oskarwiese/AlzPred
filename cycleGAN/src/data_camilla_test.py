# CAMILLAS KODE IKKE VORES

import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

from torch.utils.data import Dataset, DataLoader
import torchio as tio
import pandas as pd

# put the csv fle in your home folder with this file
df = pd.read_csv("/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/oskar_implemetation/dataset_with_filenames1.csv", engine='python', sep=',')
#df = df.loc[(df['Description'] == 'Accelerated Sagittal MPRAGE') | (df['Description'] == 'MPRAGE') | (df['Description'] == 'MP-RAGE')]
df = df.loc[(df['Group'] == 'AD') | (df['Group'] == 'CN')]
df = df.sort_values(by=['Subject', 'Acq Date'])
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
PTIDs = df['Subject'].astype('str').tolist()
ImageIDs = df['Image Data ID'].astype('str').tolist()
diagnosis = df['Group'].astype('str').tolist() #delete last elem, which is summaryoutput

#convert list of str to list of int
unique = set(diagnosis)
map = {word: i for i, word in enumerate(unique)}
diagnosis_int = [map[word] for word in diagnosis]
#print(diagnosis_int)
print("unique", unique)


paths = df['Path'].astype('str').tolist()



class mydataLoader(Dataset):
    def __init__(self, data_dir, label):
# code for making the csv file - not in use anymore 
       #filenames = []
        #for i in range(len(PTIDs)): #skip summaryoutput which is last elem
        #    f = glob.glob(data_dir + PTIDs[i] + "/" + r'smwc*' + ImageIDs[i] + r'*.nii')
         #   if (len(f) == 1):
         #       filenames.append(f[0])
         #   elif (len(f) > 1):
         #       print("Error found two images!", f)
         #       exit()
         #   else:
         #       filenames.append("not found")
#extracted from the datafile
        self.image_paths = paths
        self.labels = label
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        path = self.image_paths[index]
        label = self.labels[index]
        image = tio.ScalarImage(path)
        image = np.array(image)
        image = torch.from_numpy(image[:,:240,:256,:160]).float()
        return image, label
    
#  ../../../../dtu-compute/ADNIbias/ADNI1_baseline3T_collection/
dataset = mydataLoader(data_dir='../../../../dtu-compute/ADNIbias/spm12_ADNI1/ADNI_dartel/', label=diagnosis_int)
print(dataset)

dataloader = DataLoader(dataset, batch_size=3,
                        shuffle=True, num_workers=2)
print(dataloader)
for i, (data, target) in enumerate(dataloader):
    #print(data, '\n')
    #print(target)
    break

print(df.Path.head(1))