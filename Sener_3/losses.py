import torch
import torch.nn.functional as F 

def nll(y_pred, y_true, val=False):
    if val:
        return F.nll_loss(y_pred, y_true, size_average=False)
    else:
        return F.nll_loss(y_pred, y_true)

def get_loss(tasks):
    loss = {}
    for t in tasks:
    	loss[t] = nll;
    return loss
