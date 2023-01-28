#Python Modules
import os
import time
import random
import torch
import numpy as np
from scipy.io import savemat

#Pandas
import pandas

#PyTorch Modules
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset
from torchvision import transforms
# from loaders import Dataset_loader

#Our Modules
from model_loader import br_dnn2
from tqdm import tqdm

#sk-learn Modules
from sklearn.metrics import fbeta_score, hamming_loss
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set fixed random number seed
torch.manual_seed(42)

#beta=1 by default
beta=1

#Load the Dataset files
train_Data_file = './../Datasets/Data2/Xtrain_2021-11-25.npy'
val_Data_file = './../Datasets/Data2/Xval_2021-11-25.npy'
test_Data_file = './../Datasets/Data2/Xtest_2021-11-25.npy'
train_binlabel_file = './../Datasets/Data2/Ytrainbin_2021-11-25.npy'
val_binlabel_file = './../Datasets/Data2/Yvalbin_2021-11-25.npy'
test_binlabel_file = './../Datasets/Data2/Ytestbin_2021-11-25.npy'

#Data statistics including d and s
ntrain, d=np.load(train_Data_file).shape
s = np.load(train_binlabel_file).shape[1]

#Validation labels and number of validation points
Yval_bin = np.load(val_binlabel_file)
nval=Yval_bin.shape[0]

#Test labels and Pred labels
Ntrains=np.linspace(10,312,50)
Ntrains = np.array([int(n) for n in Ntrains])
br_metrics=np.zeros((len(Ntrains),7+s))
#training-size, fbeta, prec, rec, ml-hamm, train-time-sec, test-time-sec
# br_fbetaacc=np.zeros((len(Ntrains),2))

#Test labels and Pred labels
Ytest_bin = np.load(test_binlabel_file)
ntest=np.load(test_Data_file).shape[0]

#Numpy array for predicted labels
Ypred_bin = np.zeros(Ytest_bin.shape)

lr_folder = os.listdir('./Data2Models/')
##################################################
##################################################
##################################################
list_lr = np.logspace(-4,-2,3)
for lr in list_lr:
    if not ('lr_'+str(lr) in lr_folder):
        os.mkdir('./Data2Models/lr_'+str(lr))
    
    #Train and Test for each size in Ntrains
    for (n_idx, n) in enumerate(Ntrains):
        # print("============New Iteration================")
        # print("n_idx={}; n={}".format(n_idx,n))
        train_time_sec=0
        test_time_sec=0
        for label_idx in range(s):
            #create tensors for train and validation sets
            Xt = torch.tensor(np.float64(np.load(train_Data_file)))[:n,:]
            Yt = torch.tensor(np.float64(np.load(train_binlabel_file)))[:n,label_idx]
            Xv = torch.tensor(np.float64(np.load(val_Data_file)))
            Yv = torch.tensor(np.float64(np.load(val_binlabel_file)))[:,label_idx]
            #create Train and Validation TensorDatasets
            train_dataset = TensorDataset(Xt,Yt);
            validation_dataset = TensorDataset(Xv,Yv);
            
            #Configure the Training Model Architecture
            inp = d
            out = 1
            net = br_dnn2(inp,out);
            net = net.float();
            net.to(device)
            print("Model architecture: ", net)
            
            # Configure no. of epochs and batch-size for training
            epochs = 100
            batchsize = 128
            
            # transformation= transforms.Compose([transforms.ToTensor()])
            # train_dataset= Dataset_loader(data_file_name=train_Data_file, label_file_name = train_label_file, transform = transformation)
            # validation_dataset= Dataset_loader(data_file_name=val_Data_file, label_file_name = val_label_file, transform = transformation)
            train_loader= DataLoader(train_dataset,batch_size=batchsize,shuffle = True)
            val_loader= DataLoader(validation_dataset,batch_size=batchsize,shuffle = True)
            
            # Configure the loss criterion and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optmizer= optim.Adam(net.parameters(), lr= lr, weight_decay=1e-4)
            #weight_decay for l2 regularization
            
            ################ Start Training Loop #####################
            print("======Start Training of Tag-"+str(label_idx)+"=========")
            traintime_start=time.time()
            plot_loss = []
            plot_loss_val = []
            # plot_fbeta = []
            plot_fbeta_val = []
            for e in range(epochs):
                train_loss = 0
                validation_loss = 0
            
                # Train the Model in each epoch
                net.train()
                for data, labels in train_loader:
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
            
                # Validate the model in each epoch
                net.eval()
                ee=0;
                for data,labels in val_loader:
                    #Get the data and labels
                    data, labels = data.to(device), labels.to(device)
                    # Forward pass on validation data
                    val_outputs = net(data.float())[:,0]
                    # Compute surrogate loss
                    loss_ = criterion(val_outputs, labels)
                    #Validation loss
                    validation_loss += loss_.item()
                    #_, val_preds = torch.max(val_outputs,1)
                    #val_correct += torch.sum(val_preds == labels.data)
                    ee = ee+1
                
                #compute accuracy on validation data for tag label_idx
                #compute accuracy on validation data (for tag label_idx)
                # first generate Yjvalpred
                Xv = Xv.to(device)
                Yjvalpredscores = net(Xv.float())[:,0]
                Yjvalpred = torch.round(torch.sigmoid(Yjvalpredscores))
                Yjvalpred = Yjvalpred.cpu().detach().numpy()
                #now compute accuracy
                validation_fbeta = fbeta_score( Yval_bin[:,label_idx], Yjvalpred, 
                                beta=1)
                    
                train_loss = train_loss/len(train_dataset)
                validation_loss = validation_loss/len(validation_dataset)
                #train_acc = train_correct.double() / len(train_dataset)
                print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tValidation Fbeta {:.8f}'
                      .format(e+1, train_loss, validation_loss,
                              validation_fbeta))
                #Append the training and validation losses to the loss-plot lists
                plot_loss.append(train_loss);
                plot_loss_val.append(validation_loss);
                
                if e % 9 == 0 and n_idx==len(Ntrains)-1:
                    # Save after every certain epoch
                    torch.save(
                        net.state_dict(),'./Data2Models/lr_'+str(lr)+
                        '/tag_'+str(label_idx)+
                        '/br-{}-{}-{}-{}-{}-{:.4f}.pt'.format(
                           label_idx, 
                           n ,e+1,batchsize,lr,
                           validation_fbeta)
                        )
            traintime_end=time.time()
            train_time_sec+=(traintime_end-traintime_start)
            print("=======Training of Tag-"+str(label_idx)+" Complete======")
            ################## End Training Loop ###################
            ############## Start Testing for tag label_idx ##############
            print("========Testing of Tag-"+str(label_idx)+" Started========")
            testtime_start=time.time()
            Xtest = torch.tensor(np.float64(np.load(test_Data_file)))
            Xtest = Xtest.to(device)
            Yjtest = Ytest_bin[:,label_idx]
            Yjpredscores = net(Xtest.float())[:,0]
            Yjpred = torch.round(torch.sigmoid(Yjpredscores))
            Yjpred = Yjpred.cpu().detach().numpy()
            Ypred_bin[:,label_idx]=Yjpred
            testtime_end=time.time()
            test_time_sec+=(testtime_end-testtime_start)
            print("========Testing of Tag-"+str(label_idx)+" Complete========")
            ############# End Testing for tag label_idx #############    
        
        ##################################################
        #We are in (n_idx,n) loop here
        ##################################################
        ################ Evaluation Metrics###############
        print("======Metrics Calculation Started============") 
        #compute average precision
        prec=precision_score(Ytest_bin, Ypred_bin, average='micro')        
        #compute average recall
        rec=recall_score(Ytest_bin, Ypred_bin, average='micro')
        #compute fbeta_accuracy
        fbetascore=fbeta_score(Ytest_bin, Ypred_bin, beta=1, average='samples')
        #compute multilabel hamming loss
        ml_hamming=hamming_loss(Ytest_bin, Ypred_bin)
        #compute per-label hamming losses
        pl_hamming=np.zeros(s)
        for j in range(s):
            pl_hamming[j] = hamming_loss(Ytest_bin[:,j], Ypred_bin[:,j])
                # print("Per-label Hamming loss of label "+str(j)
                      # +": {}".format(pl_hamming[j]))
        print("F-beta score: {}".format(fbetascore))
        print("Multi-label Hamming Loss: {}".format(ml_hamming))
        print("Training Time (sec): {}".format(train_time_sec))
        print("Test Time (sec): {}".format(test_time_sec))
        br_metrics[n_idx,:]=np.concatenate(
                ( np.array([n, fbetascore, prec, rec, ml_hamming,
                            train_time_sec, test_time_sec]),
                 pl_hamming)
                )
        print("=======Metrics Calculation Complete===========")
        #save the metrics
        np.save("./../Datasets/Data2/Metrics/br_lr_"+str(lr)+
                    "_metrics_data2.npy", np.array(br_metrics))
        dictionary={'br': br_metrics}
        savemat("./../Datasets/Data2/Metrics/br_lr_"+str(lr)+
            "_metrics_data2.mat", dictionary)
        ######################### Evaluation Metrics#####################
    ##################################################
    ##################################################
    ##################################################
    ########################### Save Metrics #####################
    np.save("./../Datasets/Data2/Metrics/br_lr_"+str(lr)+
                "_metrics_data2.npy",np.array(br_metrics))
    dictionary={'br': br_metrics}
    savemat("./../Datasets/Data2/Metrics/br_lr_"+str(lr)+
            "_metrics_data2.mat", dictionary)
    #################### Save Excel File ##################
    columns=['Training-size', 'Fbeta-score', 
             'Precision','Recall',
             'ML-Hamm','Training-time (sec)', 'Test-time (sec)']
    for j in range(s):
        columns+= ['PL-Hamm-'+str(j)]
    df = pandas.DataFrame(br_metrics, columns=columns)
    df.to_excel("./../Datasets/Data2/Metrics/br_lr_"+str(lr)+
                    "_metrics_data2.xlsx" ,index=False)
    ##################################################
    ##################################################
    ##################################################    
    ########################### Save Fbeta Accuracy ########################
    # ymdstr = time.strftime("%Y-%m-%d")
    # np.save("./../Datasets/Data2/Metrics/br_metrics_"+ymdstr+".npy",np.array(br_fbetaacc))
    
