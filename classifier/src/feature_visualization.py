import classifier_model
from sklearn.manifold import TSNE
from bioinfokit.visuz import cluster
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import seaborn as sns
import utils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Test to see if GPU is available
print(device)

torch.cuda.empty_cache()

generated = True

if not generated:
    nøym = 'not_generated_imgs_'
    data = pd.read_csv('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/csv_data/alz_data_without_generated.csv')
else:
    nøym = 'with_generated_imgs_'
    data = pd.read_csv('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/csv_data/alz_data.csv')

df_train, df_test = utils.train_test_split(data)

X_train, X_test = df_train.Path.astype(str).tolist(), df_test.Path.astype(str).tolist()
y_train, y_test = df_train.Label.tolist(), df_test.Label.tolist()

dataset_train = utils.mydataLoader(X_train, y_train, 0) # Class needs paths, label, train or val as binary
dataset_val   = utils.mydataLoader(X_test,  y_test,  1)

dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4)
dataloader_val   = DataLoader(dataset_val,   batch_size=1, shuffle=True, num_workers=4)

print(f'Length of training data: {len(dataloader_train)}')
print(f'Length of validation data: {len(dataloader_val)}')

conv = classifier_model.ConvNet()
path_to_model = '/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/models/new_best-model_epoch_180.pth.tar'
conv.load_state_dict(torch.load( \
    path_to_model),
    strict=False )

conv.eval()
conv.to(device)

saved = []

for i, (val, label) in tqdm(enumerate(dataloader_val)):

    val = val.to(device)

    # Get predictions from the maximum value
    saved.append(conv(val).flatten().tolist())

X = saved # Les predictionesz

tsneeeee = TSNE(n_components=2).fit_transform(X)
cluster.tsneplot(score=tsneeeee, colorlist=np.array(df_test.Type), colordot=('#d690f8','#460664'), figname=f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/plots/{nøym}tsne_type', dim = (10, 7))
cluster.tsneplot(score=tsneeeee, colorlist=np.array(df_test.Group), colordot=('#d690f8','#460664'), figname=f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/plots/{nøym}tsne_group', dim = (10, 7))
cluster.tsneplot(score=tsneeeee, colorlist=np.array(df_test.Sex), colordot=('#d690f8','#460664'), figname=f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/plots/{nøym}tsne_sex', dim = (10, 7))



pca = PCA(n_components=2)
pca.fit(X)
pca = pca.transform(X)
plt.figure(figsize = (10,7))
sns.scatterplot(pca[:, 0], pca[:, 1], s=70, hue = df_test.Type, palette = ['#d690f8','#460664'])
plt.title('2D Scatterplot of Principle Component 1 & 2')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/plots/{nøym}pca_type.png')
plt.clf()
plt.cla()
plt.close()
plt.figure(figsize = (10,7))
sns.scatterplot(pca[:, 0], pca[:, 1], s=70, hue = df_test.Type, palette = ['#d690f8','#460664'])
plt.title('2D Scatterplot of Principle Component 1 & 2')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/plots/{nøym}pca_group.png')
plt.clf()
plt.cla()
plt.close()
plt.figure(figsize = (10,7))
sns.scatterplot(pca[:, 0], pca[:, 1], s=70, hue = df_test.Type, palette = ['#d690f8','#460664'])
plt.title('2D Scatterplot of Principle Component 1 & 2')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/plots/{nøym}pca_sex.png')
plt.clf()
plt.cla()
plt.close()



# TODO: implement with Alzheimer labels as well 

# TODO: look at age or sex