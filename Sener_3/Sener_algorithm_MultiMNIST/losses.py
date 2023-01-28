import torch
import torch.nn.functional as F 

def nll(y_pred, y_true, val=False):
    if val:
        return F.nll_loss(y_pred, y_true, size_average=False)
    else:
        return F.nll_loss(y_pred, y_true)

def get_loss(tasks):
    loss = {}
    loss[tasks[0]] = nll;
    loss[tasks[1]] = nll;
    return loss
