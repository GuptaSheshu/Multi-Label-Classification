#Python Modules
import os
import time
import random
import torch
import numpy as np

#PyTorch Modules
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset
from torchvision import transforms
# from loaders import Dataset_loader

#Our Modules
from model_loader import br_dnn
from tqdm import tqdm

#sk-learn Modules
from sklearn.metrics import hamming_loss
from sklearn.metrics import fbeta_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set fixed random number seed
torch.manual_seed(42)

#beta=1 by default
beta=1

#Load the Dataset files
train_Data_file = './Data3/.npy'
val_Data_file = './Data3/.npy'
test_Data_file = './Data3/.npy'
train_binlabel_file = './Data3/.npy'
val_binlabel_file = './Data3/.npy'
test_binlabel_file = './Data3/.npy'

#Data statistics including d and s
ntrain, d=np.load(train_Data_file).shape
s = np.load(train_binlabel_file).shape[1]
ntest=np.load(test_Data_file).shape[0]
Yval_bin = np.load(val_binlabel_file)
nval=Yval_bin.shape[0]

# Ntrains=np.linspace(10,10,1)
# Ntrains = np.array([int(n) for n in Ntrains])
# br_acc=np.zeros(len(Ntrains))

#Test labels and Pred labels
Ytest_bin = np.load(test_binlabel_file)
Ypred_bin = np.zeros(Ytest_bin.shape)

#Train a Binary classifier for each tag and perform its test on Xtest also
for label_idx in range(s):
    #create tensors for train and validation sets
    Xt = torch.tensor(np.float64(np.load(train_Data_file)))
    Yt = torch.tensor(np.float64(np.load(train_binlabel_file)))[:,label_idx]
    Xv = torch.tensor(np.float64(np.load(val_Data_file)))
    Yv = torch.tensor(np.float64(np.load(val_binlabel_file)))[:,label_idx]
    
    #create Train and Validation TensorDatasets
    train_dataset = TensorDataset(Xt,Yt);
    validation_dataset = TensorDataset(Xv,Yv);
    
    #Configure the Training Model Architecture
    inp = d
    out = 1
    net = br_dnn(inp,out);
    net = net.float();
    net.to(device)
    print("Model architecture: ", net)
    
    # Configure no. of epochs, learning-rate, and batch-size for training
    epochs = 1 
    batchsize = 128
    lr = 0.001 #learning rate
    
    # transformation= transforms.Compose([transforms.ToTensor()])
    # train_dataset= Dataset_loader(data_file_name=train_Data_file, label_file_name = train_label_file, transform = transformation)
    # validation_dataset= Dataset_loader(data_file_name=val_Data_file, label_file_name = val_label_file, transform = transformation)
    train_loader= DataLoader(train_dataset,batch_size=batchsize,shuffle = True)
    val_loader= DataLoader(validation_dataset,batch_size=batchsize,shuffle = True)
    
    # Configure the loss criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optmizer= optim.Adam(net.parameters(), lr= lr, weight_decay=1e-4)
    #weight_decay for l2 regularization
    
    ############################# Start Training Loop ############################
    print("==========================Start Training=========================")
    plot_loss = []
    plot_loss_val = []
    for e in range(epochs):
        train_loss = 0
        validation_loss = 0
    
        # Train the Model in each epoch
        net.train()
        for data, labels in tqdm(train_loader):
            #Get the data and labels
            data, labels = data.to(device), labels.to(device)
            # Zero the gradients
            optmizer.zero_grad()
            # Forward pass on train data
            outputs = net(data.float())[:,0]
            # Compute BCE with Logit Loss
            loss = criterion(outputs, labels)
            # Backward Pass
            loss.backward()
            # Perform Optimization
            optmizer.step()  # W = W - eta*grad;
            #Training loss
            train_loss += loss.item()
            #, preds = torch.max(outputs,1)
            #train_correct += torch.sum( preds == labels.data )
    
        # validate the model in each epoch
        net.eval()
        ee=0;
        for data,labels in tqdm(val_loader):
            #Get the data and labels
            data, labels = data.to(device), labels.to(device)
            # Forward pass on validation data
            val_outputs = net(data.float())[:,0]
            # Compute Zhang's surrogate loss
            loss_ = criterion(val_outputs, labels)
            #Validation loss
            validation_loss += loss_.item()
            #_, val_preds = torch.max(val_outputs,1)
            #val_correct += torch.sum(val_preds == labels.data)
            ee = ee+1
            
        train_loss = train_loss/len(train_dataset)
        validation_loss = validation_loss/len(validation_dataset)
        #train_acc = train_correct.double() / len(train_dataset)
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} '
              .format(e+1, train_loss, validation_loss))
        #Append the training and validation losses to the loss-plot lists
        plot_loss.append(train_loss);
        plot_loss_val.append(validation_loss);
    
    #Save the learned model
    torch.save(net.state_dict(),'./Data3Models/br-{}-{}-{}-{}.pt'.format(
        label_idx,epochs,batchsize,lr)
        )
    print("==========================End Training=========================")
    ############################# End Training Loop #########################
    
    ################ Start Testing for tag label_idx ########################
    #Perform Testing using the Trained Model for tag label_idx
    print("==========================Testing Started=========================")
    Xtest = torch.tensor(np.float64(np.load(test_Data_file)))
    Xtest = Xtest.to(device)
    Yjtest = Ytest_bin[:,label_idx]
    
    Yjpredscores = net(Xtest.float())[:,0]
    Yjpred = torch.round(torch.sigmoid(Yjpredscores))
    Yjpred = Yjpred.cpu().detach().numpy()
    
    # Yvalpredscores2 = net(Xv.float())
    # Yvalpredscores2 = Yvalpredscores2.cpu().detach.numpy()
    Ypred_bin[:,label_idx]=Yjpred
    print("========================Testing Complete========================")
    ################ End Testing for tag label_idx ########################
    
##################################################
##################################################
##################################################
########################### Evaluation Metrics########################
print("=======================Evaluation Started========================") 
#compute fbeta_accuracy
fbetascore=fbeta_score(Ytest_bin, Ypred_bin, beta=1, average='samples')
print("F-beta score: {}".format(fbetascore))
br_fbetaacc=fbetascore
# print(fbetascore=fbeta_score(Yval_bin, Yvalpred_bin, beta=1, average='samples'))
#compute hammingloss
mlhammingloss=hamming_loss(Ytest_bin, Ypred_bin)
print("Multilabel Hamming Loss: {}".format(mlhammingloss))
#compute per-label hamming loss
for j in range(s):
    plhammingloss = hamming_loss(Ytest_bin[:,j], Ypred_bin[:,j])
    print("Per-label Hamming loss of label "+str(j)
          +": {}".format(plhammingloss))
#compute ranking_loss
print("=======================Evaluation Complete========================")
########################### Evaluation Metrics########################

##################################################
##################################################
##################################################
########################### Save Fbeta Accuracy ########################
ymdstr = time.strftime("%Y-%m-%d")
np.save("Data3/br_fbetaacc_data3_"+ymdstr+".npy",np.array(br_fbetaacc))
