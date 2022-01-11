import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

print(torch.__version__)
dataset = TensorDataset(torch.randn(100, 1), torch.randn(100, 1))
loader = DataLoader(dataset, batch_size=25, num_workers=2, persistent_workers=True)

for data, target in loader:
    print(data.shape)
