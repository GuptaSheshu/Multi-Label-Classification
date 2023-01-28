import torch
import torch.nn as nn
import torch.nn.functional as F


class br_dnn1(nn.Module):
    def __init__(self,inp,out):
        super(br_dnn1,self).__init__()
        self.out = out;
        self.inp = inp;

        self.dnn = nn.Sequential(
            nn.Linear(self.inp, self.out)
        )

    def forward(self,x):
        x = self.dnn(x);
        return x
    

class br_dnn2(nn.Module):
    def __init__(self,inp,out):
        super(br_dnn2,self).__init__()
        self.out = out;
        self.inp = inp;
        #dataset-2 has self.inp=72 features and 6 labels
        #72>>48>>24>>1
        self.dnn = nn.Sequential(
            #linear - no hidden layer
            nn.Linear(self.inp, self.out)
            #1 hidden layer
            # nn.Linear(self.inp, 36),
            # nn.ReLU(),
            # nn.Linear(36,self.out)
        )

    def forward(self,x):
        x = self.dnn(x);
        return x;