import torch as tr
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_a(y,j,k): #j and k denote indices wrt Fbeta paper
    sum_row1 = tr.sum(tr.abs(y),1);
    sum_row2 = tr.sum(y,1)
    # if sum_row2!=sum_row1:
    #     print("Something is wrong.")
    sum_row=sum_row2
    
    temp = (sum_row==k).type(tr.float64);
    if j==0:
        return temp;
    return tr.mul(temp,y[:,j-1]);


def phi(t,y): #t=1 or -1 y are 2D tensors
	return tr.log(1+tr.exp(-t*y));

def fbeta_loss(y_pred,y_true):
    ayo = get_a(y_true,0,0)
    '''
    Compute Average Fbeta loss over y_pred and y_true tensors
    '''
    intial_term = tr.mul(ayo,phi(1,y_pred[:,0])) + tr.mul((1-ayo),phi(-1,y_pred[:,0]))
    temp = tr.ones(y_pred.shape[0],y_pred.shape[1]-1)
    temp = temp.to(device)
    ct=0
    for j in range(1,y_true.shape[1]+1): #these +1s are for proper indexing
        for k in range(1, y_true.shape[1]+1):
            temp[:,ct] = get_a(y_true,j,k)
			#temp = get_a(y_true,j,k+1);
            ct+=1
    second_term = tr.mul(temp,phi(1,y_pred[:,1:])) + tr.mul((1-temp),phi(-1,y_pred[:,1:]))
    second_term_sum = tr.sum(second_term,1)
    # if tr.Tensor.size(second_term_sum_prob_wrong)[0]>1:
        # print("Yes, it should be wrong.")
    # second_term_sum2 = tr.sum(second_term)
    # second_term_sum=second_term_sum2

    loss_vec = intial_term+second_term_sum
    # print(loss_vec.shape)
    loss = tr.mean(loss_vec)
    # loss = tr.sum(loss_vec)
    return loss


