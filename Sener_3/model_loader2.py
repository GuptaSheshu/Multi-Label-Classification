import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#for d=72, s=6
class SharedModel(nn.Module):
    def __init__(self,inp, thetash):
        super(SharedModel, self).__init__()
        self.dnn = nn.Sequential(
                nn.Linear(inp,thetash),
                nn.ReLU()
                )

    def forward(self, x):
        x = self.dnn(x);
        return x


class TaskModel(nn.Module):
    def __init__(self,out, thetash):
        super(TaskModel,self).__init__()

        self.dnn = nn.Sequential(
            nn.Linear(thetash, out),
        )

    def forward(self,x):
        x = F.log_softmax(self.dnn(x),dim=1);
        return x;

def get_model(parallel,tasks,inp,out):
    thetash=6
    model = {}
    model['rep'] = SharedModel(inp, thetash)

    for t in tasks:
        model[t] = TaskModel(out, thetash)

    if parallel:
        model['rep'] = nn.DataParallel(model['rep'])
        for t in tasks:
            model[t] = nn.DataParallel(model[t])        
            
    return model

