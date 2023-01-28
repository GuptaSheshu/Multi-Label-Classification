import torch
import torch.nn as nn
import torch.nn.functional as F


class fbeta_dnn1(nn.Module):
    def __init__(self,inp,out):
        super(fbeta_dnn1,self).__init__()
        self.out = out;
        self.inp = inp;

        self.dnn = nn.Sequential(
            nn.Linear(self.inp, self.out)
        )

    def forward(self,x):
        x = self.dnn(x);
        return x;


class fbeta_dnn2(nn.Module):
    def __init__(self,inp,out):
        super(fbeta_dnn2,self).__init__()
        self.out = out;
        self.inp = inp;
        #dataset-2 has self.inp=72 features and 37 outputs
        self.dnn = nn.Sequential(
            #linear - no hidden layer
            nn.Linear(self.inp, 72),
            nn.ReLU(),
            nn.Linear(72,self.out)
            #1 hidden layer
            # nn.Linear(self.inp, self.inp),
            # nn.ReLU(),
            # nn.Linear(self.inp, self.out)
            #2 hidden layes
            # nn.Linear(self.inp, self.inp),
            # nn.ReLU(),
            # nn.Linear(self.inp, self.inp),
            # nn.ReLU(),
            # nn.Linear(self.inp, self.out)
        )
        
    def forward(self,x):
        x = self.dnn(x);
        return x;

class fbeta_dnn3(nn.Module):
    def __init__(self,inp,out):
        super(fbeta_dnn3,self).__init__()
        self.out = out;
        self.inp = inp;

        self.dnn = nn.Sequential(
            nn.Linear(self.inp, self.out)
        )

    def forward(self,x):
        x = self.dnn(x);
        return x;