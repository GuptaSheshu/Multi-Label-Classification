import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pywt
import os
import math 

class Dataset_loader(Dataset):
    def __init__(self,data_file_name,label_file_name,transform):
        '''
        Pytorch Dataset class
        params:-
                 file_name : the path of the npy file    (train, validation, test)
        return :-
                 XTrain, YTrain
        '''
        self.data = np.load(data_file_name);
        self.labels = np.load(label_file_name);
        self.transform = transform;

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datas = np.array(self.data[idx].reshape(-1,1))
        labless = np.array(self.labels[idx])
        labless = torch.from_numpy(labless).long()
        if self.transform :
            datas = self.transform(datas)
        
        return datas,labless

