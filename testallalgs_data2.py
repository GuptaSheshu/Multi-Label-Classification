#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 22:44:11 2021

@author: nightstalker
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:08:31 2021

@author: khann
"""
#Python Modules
import numpy as np
from scipy.io import savemat
import time
#Pandas
import pandas
import torch
from sklearn.metrics import fbeta_score, hamming_loss
from sklearn.metrics import precision_score, recall_score
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils import data
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from Sener_3.loaders2 import Dataset_loader

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

## Dataset loader
beta=1

#Load the Dataset files
test_Data_file = './Datasets/Data2/Xtest_2021-12-07.npy'
test_binlabel_file = './Datasets/Data2/Ytestbin_2021-12-07.npy'
Ptest_file = './Datasets/Data2/Ptest_2021-12-07.npy'

#Test labels and number of test points
Ytest_bin = np.load(test_binlabel_file)
ntest=Ytest_bin.shape[0]
Ptest = np.load(Ptest_file)
Ypred_bin = np.zeros(Ytest_bin.shape)
###################################################
###################################################
###################################################
################# Bayes Optimal Classifier #####################
#test and compute bayes optimal classifier's Fbeta accuracy
# s=10
# beta=1
# Fbeta=np.zeros((2**s,2**s))
# for yhat in range(2**s): #yhats are along the rows
#     for yt in range(2**s): #ys are along the columns
#         if yhat==0 and yt==0:
#             Fbeta[yhat][yt]=1
#         else:
#             temp=bin(yhat).replace("0b","").zfill(s)
#             yhatbin=np.array([int(c) for c in temp])
#             temp=bin(yt).replace("0b","").zfill(s)
#             ytbin=np.array([int(c) for c in temp]) 
#             Fbeta[yhat][yt]=(1+beta**2)*np.dot(ytbin,yhatbin)/(
#                 (beta**2)*np.sum(ytbin)+np.sum(yhatbin)
#                 )
# E=Fbeta@(Ptest.transpose())
# Ypred=np.argmax(E,0)
# Ypred_bin=np.zeros((ntest,s))
# for i in range(len(Ypred)):
#     temp=bin(Ypred[i]).replace("0b","").zfill(s)
#     Ypred_bin[i,:]=np.array([int(c) for c in temp])
# #compute accuracy of Bayes optimal classifier
# bayes_fbeta=0
# for i in range(ntest):
#     ytestbin=Ytest_bin[i,:]
#     ypredbin=Ypred_bin[i,:]
#     temp=(1+beta**2)*np.dot(ytestbin,ypredbin)/(
#         (beta**2)*np.sum(ytestbin)+np.sum(ypredbin)
#         )
#     bayes_fbeta+=temp
# bayes_fbeta=bayes_fbeta/ntest
# print("\nBayes-Optimal Classifier's F-beta score: {}".format(bayes_fbeta))

###################################################
###################################################
###################################################
################# Zhang's best model #####################
print("\n\nZhang metrics")
from Zhang_2.model_loader import fbeta_dnn3
from Zhang_2.decode import decode
d=200
s=10
inp = d
out = s**2+1
net = fbeta_dnn3(inp,out);
net.load_state_dict(torch.load("./FinalModels/Zhang/Data2Models/"+
                               "zhang-7000-100-128-0.001-0.6490.pt"))
Xtest = torch.tensor(np.float64(np.load(test_Data_file)));
Xtest = Xtest.to(device)

testtime_start=time.time()
Ypredscores = net(Xtest.float());
Ypredscores = Ypredscores.cpu().detach().numpy()
for i in range(Ypredscores.shape[0]):
    Ypred_bin[i,:] = decode(Ypredscores[i], beta=1)
testtime_end=time.time()
test_time_sec=(testtime_end-testtime_start)

prec=precision_score(Ytest_bin, Ypred_bin, average='micro')        
rec=recall_score(Ytest_bin, Ypred_bin, average='micro') 
fbetascore=fbeta_score(Ytest_bin, Ypred_bin, 
                       beta=1, average='samples')
ml_hamming=hamming_loss(Ytest_bin, Ypred_bin)
pl_hamming=np.zeros(s)

print("F-beta score: {}".format(fbetascore))
print("Precision: {}".format(prec))
print("Recall: {}".format(rec))
print("Multi-label Hamming Loss: {}".format(ml_hamming))
for j in range(s):
    pl_hamming[j] = hamming_loss(Ytest_bin[:,j], Ypred_bin[:,j])
    print("Per Task Hamming Loss Task-{} :{}".format(j,pl_hamming[j]))
print("Test Time (sec): {}".format(test_time_sec))

###################################################
###################################################
###################################################
################# Sener's Best Model #################
tasks = ['0','1','2','3','4','5','6','7','8','9'];
inp = d
out = 2
###################################################
###################################################
###################################################
###################Dnn1
print("\n\nSener1 metrics :")
from Sener_3.model_loader3 import get_model
dic = torch.load("./FinalModels/Sener/Data2Models/"+
                 "sener_dnn1-7000-91-128-0.001-0.4726.pt")
net = get_model(parallel=False, tasks=tasks,inp=inp,out=2);
net['rep'].load_state_dict(dic['model_rep'])
for t in tasks:
    net[t].load_state_dict(dic['model_{}'.format(t)])

transformation = transforms.Compose([transforms.ToTensor()])
test_dataset = Dataset_loader(data_file_name = test_Data_file,
                              label_file_name = test_binlabel_file, 
                              transform = transformation,n=ntest)
test_loader= DataLoader(test_dataset, batch_size=ntest,
                        shuffle = False)

Ypred_bin = np.zeros(Ytest_bin.shape)
testtime_start=time.time()
for batch_test in test_loader:
    val_images = batch_test[0];
    labels_val={}
    for i, t in enumerate(tasks):
        if t not in tasks:
            continue
        labels_val[t] = batch_test[i+1]
    val_rep = net['rep'](val_images.float())
    for t in tasks:
        out_t_val = net[t](val_rep)
        ypred = np.argmax(out_t_val.detach().numpy(),axis=1);
        Ypred_bin[:,int(t)] = ypred
testtime_end=time.time()
test_time_sec=(testtime_end-testtime_start)

prec=precision_score(Ytest_bin, Ypred_bin, average='micro')        
rec=recall_score(Ytest_bin, Ypred_bin, average='micro') 
fbetascore=fbeta_score(Ytest_bin, Ypred_bin, 
                       beta=1, average='samples')
ml_hamming=hamming_loss(Ytest_bin, Ypred_bin)
pl_hamming=np.zeros(s)

print("F-beta score: {}".format(fbetascore))
print("Precision: {}".format(prec))
print("Recall: {}".format(rec))
print("Multi-label Hamming Loss: {}".format(ml_hamming))
for j in range(s):
    pl_hamming[j] = hamming_loss(Ytest_bin[:,j], Ypred_bin[:,j])
    print("Per Task Hamming Loss Task-{} :{}".format(j,pl_hamming[j]))
print("Test Time (sec): {}".format(test_time_sec))

###################################################
###################################################
###################################################
###################Dnn2
print("\n\nSener2 metrics :")
#Dnn2
from Sener_3.model_loader3_dnn2 import get_model
dic = torch.load("./FinalModels/Sener/Data2Models/"+
                 "sener_dnn2-7000-91-128-0.001-0.4963.pt")
net = get_model(parallel=False, tasks=tasks,inp=inp,out=2);
net['rep'].load_state_dict(dic['model_rep'])
for t in tasks:
    net[t].load_state_dict(dic['model_{}'.format(t)])
transformation = transforms.Compose([transforms.ToTensor()])
test_dataset = Dataset_loader(data_file_name = test_Data_file,
                              label_file_name = test_binlabel_file, 
                              transform = transformation,n=ntest)
test_loader= DataLoader(test_dataset, batch_size=ntest,
                        shuffle = False)

Ypred_bin = np.zeros(Ytest_bin.shape)
testtime_start=time.time()
for batch_test in test_loader:
    val_images = batch_test[0]
    labels_val={}
    for i, t in enumerate(tasks):
        if t not in tasks:
            continue
        labels_val[t] = batch_test[i+1]
    val_rep = net['rep'](val_images.float())
    for t in tasks:
        out_t_val = net[t](val_rep)
        ypred = np.argmax(out_t_val.detach().numpy(),axis=1);
        Ypred_bin[:,int(t)] = ypred
testtime_end=time.time()
test_time_sec=(testtime_end-testtime_start)

prec=precision_score(Ytest_bin, Ypred_bin, average='micro')        
rec=recall_score(Ytest_bin, Ypred_bin, average='micro') 
fbetascore=fbeta_score(Ytest_bin, Ypred_bin, 
                       beta=1, average='samples')
ml_hamming=hamming_loss(Ytest_bin, Ypred_bin)
pl_hamming=np.zeros(s)
print("F-beta score: {}".format(fbetascore))
print("Precision: {}".format(prec))
print("Recall: {}".format(rec))
print("Multi-label Hamming Loss: {}".format(ml_hamming))
for j in range(s):
    pl_hamming[j] = hamming_loss(Ytest_bin[:,j], Ypred_bin[:,j])
    print("Per Task Hamming Loss Task-{} :{}".format(j,pl_hamming[j]))
print("Test Time (sec): {}".format(test_time_sec))
###################

###################################################
###################################################
###################################################
###################Dnn3
print("\n\nSener3 metrics:")
from Sener_3.model_loader3_dnn3 import get_model
dic = torch.load("./FinalModels/Sener/Data2Models/"+
                 "sener_dnn3-7000-100-128-0.001-0.5130.pt")
net = get_model(parallel=False, tasks=tasks,inp=inp,out=2);
net['rep'].load_state_dict(dic['model_rep'])
for t in tasks:
    net[t].load_state_dict(dic['model_{}'.format(t)])
transformation = transforms.Compose([transforms.ToTensor()])
test_dataset = Dataset_loader(data_file_name = test_Data_file,
                              label_file_name = test_binlabel_file, 
                              transform = transformation,n=ntest)
test_loader= DataLoader(test_dataset, batch_size=ntest,
                        shuffle = False)

Ypred_bin = np.zeros(Ytest_bin.shape)
testtime_start=time.time()
for batch_test in test_loader:
    val_images = batch_test[0];
    labels_val={}
    for i, t in enumerate(tasks):
        if t not in tasks:
            continue
        labels_val[t] = batch_test[i+1]
    val_rep = net['rep'](val_images.float())
    for t in tasks:
        out_t_val = net[t](val_rep)
        ypred = np.argmax(out_t_val.detach().numpy(),axis=1);
        Ypred_bin[:,int(t)] = ypred
testtime_end=time.time()
test_time_sec=(testtime_end-testtime_start)

prec=precision_score(Ytest_bin, Ypred_bin, average='micro')        
rec=recall_score(Ytest_bin, Ypred_bin, average='micro') 
fbetascore=fbeta_score(Ytest_bin, Ypred_bin, 
                       beta=1, average='samples')
ml_hamming=hamming_loss(Ytest_bin, Ypred_bin)
pl_hamming=np.zeros(s)
print("F-beta score: {}".format(fbetascore))
print("Precision: {}".format(prec))
print("Recall: {}".format(rec))
print("Multi-label Hamming Loss: {}".format(ml_hamming))
for j in range(s):
    pl_hamming[j] = hamming_loss(Ytest_bin[:,j], Ypred_bin[:,j])
    print("Per Task Hamming Loss Task-{} :{}".format(j,pl_hamming[j]))
print("Test Time (sec): {}".format(test_time_sec))

###################################################
###################################################
###################################################
################# BR's Best Model #################
print("\n\nBR metrics:")
from BR_1.model_loader import br_dnn1
Xtest = torch.tensor(np.float64(np.load(test_Data_file)))
Xtest = Xtest.to(device)
Ypred_bin = np.zeros(Ytest_bin.shape)
file_name = ["br-0-7000-100-128-0.01-0.5794.pt",
             "br-1-7000-82-128-0.01-0.5504.pt",
             "br-2-7000-91-128-0.01-0.6572.pt",
             "br-3-7000-100-128-0.01-0.6205.pt",
             "br-4-7000-82-128-0.01-0.6134.pt",
             "br-5-7000-82-128-0.01-0.6341.pt",
             "br-6-7000-100-128-0.01-0.5569.pt",
             "br-7-7000-82-128-0.01-0.6635.pt",
             "br-8-7000-100-128-0.01-0.5868.pt",
             "br-9-7000-82-128-0.01-0.6847.pt"]

testtime_start=time.time()
for label_idx in range(s):
    Yjtest = Ytest_bin[:,label_idx]
    net = br_dnn1(inp,1);
    net.load_state_dict(torch.load("./FinalModels/BR/Data2Models/"+
                                   file_name[label_idx],map_location=torch.device('cpu')))
    Yjpredscores = net(Xtest.float())[:,0]
    Yjpred = torch.round(torch.sigmoid(Yjpredscores))
    Yjpred = Yjpred.cpu().detach().numpy()
    Ypred_bin[:,label_idx]=Yjpred

testtime_end=time.time()
test_time_sec=(testtime_end-testtime_start)

prec=precision_score(Ytest_bin, Ypred_bin, average='micro')        
rec=recall_score(Ytest_bin, Ypred_bin, average='micro')
fbetascore=fbeta_score(Ytest_bin, Ypred_bin, beta=1, average='samples')
ml_hamming=hamming_loss(Ytest_bin, Ypred_bin)
pl_hamming=np.zeros(s)
print("F-beta score: {}".format(fbetascore))
print("Precision: {}".format(prec))
print("Recall: {}".format(rec))
print("Multi-label Hamming Loss: {}".format(ml_hamming))
for j in range(s):
    pl_hamming[j] = hamming_loss(Ytest_bin[:,j], Ypred_bin[:,j])
    print("Per Task Hamming Loss Task-{} :{}".format(j,pl_hamming[j]))
print("Test Time (sec): {}".format(test_time_sec))
