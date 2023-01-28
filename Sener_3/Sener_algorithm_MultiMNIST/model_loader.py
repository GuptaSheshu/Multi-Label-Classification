import torch.nn as nn
import torch
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiLeNetRep(nn.Module):
    def __init__(self):
        super(MultiLeNetRep, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(320, 50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        x = self.dropout(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc(x))
        return x


class MultiLeNetO(nn.Module):
    def __init__(self):
        super(MultiLeNetO, self).__init__()
        self.fc1 = nn.Linear(50, 50)
        self.dropout = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        out = F.log_softmax(x,dim=1);
        return out


def get_model(dataset_name,parallel):
    if(dataset_name=='Multi_MNIST'):
        model = {}
        model['rep'] = MultiLeNetRep()
        model['L'] = MultiLeNetO()
        model['R'] = MultiLeNetO()

        if parallel:
            model['rep'] = nn.DataParallel(model['rep'])
            model['L'] = nn.DataParallel(model['L'])        
            model['R'] = nn.DataParallel(model['R'])
            
        return model

