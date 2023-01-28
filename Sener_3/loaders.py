import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Dataset_loader(Dataset):
    def __init__(self, data_file_name,label_file_name,transform,n):
        '''
        Pytorch Dataset class
        params:-
                 csv_file : the path of the csv file    (train, validation, test)
                 img_dir  : the directory of the images (train, validation, test)
                 datatype : string for searching along the image_dir (train, val, test)
                 transform: pytorch transformation over the data
        return :-
                 image, labels
        '''
        
        self.data = np.load(data_file_name)[:n,:];
        self.labels = np.load(label_file_name)[:n,:];
        self.transform = transform;

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
#        if torch.is_tensor(idx):
#            idx = idx.tolist()
#        
        datas = np.array(self.data[idx])
        labless = np.array(self.labels[idx])
        labless = torch.from_numpy(labless).long()
        if self.transform :
            datas = torch.tensor(datas)
            #datas = self.transform(datas)
        out = []
        out.append(np.asarray(datas))
        for i in range(6):
            out.append(labless[i])
        return out
        
