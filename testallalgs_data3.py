# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:08:31 2021

@author: khann
"""
#Python Modules
import numpy as np
from scipy.io import savemat

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
from Sener_3.loaders import Dataset_loader
import time

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

## Dataset loader
beta=1

#Load the TestData files
test_Data_file = './Datasets/Data3/Xtest_2021-11-25.npy'
test_binlabel_file = './Datasets/Data3/Ytestbin_2021-11-25.npy'

#Test labels and number of test points
testsize=202
Ytest_bin = np.load(test_binlabel_file)[:testsize,:]
Ypred_bin = np.zeros(Ytest_bin.shape)
ntest=Ytest_bin.shape[0]

###################################################
###################################################
###################################################
################# Zhang's best model #####################
print("\n\nZhang metrics")
from Zhang_2.model_loader import fbeta_dnn2
from Zhang_2.decode import decode
d=72
s=6
inp = d
out = s**2+1
net = fbeta_dnn2(inp,out);
#################
net.load_state_dict(torch.load("./FinalModels/Zhang/Data3Models/"+
                               "zhang-312-91-128-0.01-0.6579.pt"))
#################
Xtest = torch.tensor(np.float64(
    np.load(test_Data_file)[:testsize,:]
                                ));
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
tasks = ['0','1','2','3','4','5'];
inp = d
out = 2
###################################################
###################################################
###################################################
###################Dnn1
print("\n\nSener1 metrics :")
from Sener_3.model_loader2 import get_model
dic = torch.load("./FinalModels/Sener/Data3Models/"+
                 "sener_dnn1-312-100-128-0.01-0.2999.pt")
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
###################

###################################################
###################################################
###################################################
###################Dnn2
print("\n\nSener2 metrics :")
from Sener_3.model_loader2_dnn2 import get_model
#dic = torch.load("./Sener_3/Data1Models/lr_0.01/sener_dnn2-5000-82-128-0.01-0.6403.pt")
#dic = torch.load("./Sener_3/Data1Models/lr_0.0001/sener_dnn2-10000-100-128-0.0001-0.6132.pt")
dic = torch.load("./FinalModels/Sener/Data3Models/"+
                 "sener_dnn2-312-100-128-0.0001-0.1843.pt")
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
###################

###################################################
###################################################
###################################################
###################Dnn3
print("\n\nSener3 metrics:")
from Sener_3.model_loader2_dnn3 import get_model
dic = torch.load("./FinalModels/Sener/Data3Models/"+
                 "sener_dnn3-312-82-128-0.01-0.6033.pt")
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
Xtest = torch.tensor(np.float64(
    np.load(test_Data_file)[:testsize,:]
    ))
Xtest = Xtest.to(device)
file_name = ["br-0-312-82-128-0.01-0.4706.pt",
             "br-1-312-82-128-0.01-0.3030.pt",
             "br-2-312-100-128-0.01-0.6429.pt",
             "br-3-312-100-128-0.01-0.8772.pt",
             "br-4-312-82-128-0.01-0.6842.pt",
             "br-5-312-91-128-0.01-0.6286.pt"]
Ypred_bin = np.zeros(Ytest_bin.shape)

testtime_start=time.time()
for label_idx in range(s):
    Yjtest = Ytest_bin[:,label_idx]
    net = br_dnn1(inp,1);
    net.load_state_dict(torch.load("./FinalModels/BR/Data3Models/"+
                                   file_name[label_idx],
                                   map_location=torch.device('cpu')
                                   ))
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