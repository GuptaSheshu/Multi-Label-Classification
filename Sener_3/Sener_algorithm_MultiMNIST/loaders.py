import torch
from torchvision import transforms
from multi_mnist_loader import MNIST



def get_dataset(dataset_name,transforms,dataset_path,batchsize,download):
    if 'Multi_MNIST' in dataset_name:
        train_dst = MNIST(root=dataset_path, train=True, transform=transforms, multi=True,download=download)
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batchsize, shuffle=True, num_workers=4)

        val_dst = MNIST(root=dataset_path, train=False, transform=transforms, multi=True)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100, shuffle=True, num_workers=4)
        return train_loader, train_dst, val_loader, val_dst
