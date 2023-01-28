#Python Modules
import os
import time
import random
import numpy as np
from tqdm import tqdm
from scipy.io import savemat

#Pandas
import pandas

#PyTorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset
from torchvision import transforms
# from loaders import Dataset_loader

#Our Modules
from model_loader import fbeta_dnn1
from losses import fbeta_loss
from decode import decode

#sk-learn Modules
from sklearn.metrics import fbeta_score, hamming_loss
from sklearn.metrics import precision_score, recall_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set fixed random number seed
torch.manual_seed(42)

#beta=1 by default
beta=1

#Load the Dataset files
train_Data_file = './../Datasets/Data1/Xtrain_2021-11-13.npy'
val_Data_file = './../Datasets/Data1/Xval_2021-11-13.npy'
test_Data_file = './../Datasets/Data1/Xtest_2021-11-13.npy'
train_binlabel_file = './../Datasets/Data1/Ytrainbin_2021-11-13.npy'
val_binlabel_file = './../Datasets/Data1/Yvalbin_2021-11-13.npy'
test_binlabel_file = './../Datasets/Data1/Ytestbin_2021-11-13.npy'

#Training Data statistics including d and s
trainsize=5000
ntrain, d=np.load(train_Data_file)[:trainsize,:].shape
s = np.load(train_binlabel_file).shape[1]

#Validation labels and number of validation points
valsize=1000
Yval_bin = np.load(val_binlabel_file)[:valsize,:];
nval=Yval_bin.shape[0]

#Test labels and number of test points
testsize=3000
Ytest_bin = np.load(test_binlabel_file)[:testsize,:]
ntest=Ytest_bin.shape[0]

#Numpy array for predicted labels
Ypred_bin = np.zeros(Ytest_bin.shape)

#Ntrains is the array of increasing training-sizes
pts=50
Ntrains=np.linspace(ntrain,ntrain,1)
# Ntrains = np.linspace(4592,5000,5)
Ntrains = np.array([int(n) for n in Ntrains])
zhang_metrics=np.zeros((len(Ntrains),7+s))
#training-size, fbeta, prec, rec, ml-hamm, train-time-sec, test-time-sec

list_lr = np.logspace(-2,-2,1)
wtdecay=1e-4
#Training Data statistics including d and s
# ntrain, d=np.load(train_Data_file).shape
# s = np.load(train_binlabel_file).shape[1]

#Validation labels and number of validation points
# Yval_bin = np.load(val_binlabel_file);
# nval=Yval_bin.shape[0]

#Test labels and number of test points
# Ytest_bin = np.load(test_binlabel_file)
# ntest=np.load(test_Data_file).shape[0]

#Numpy array for predicted labels
# Ypred_bin = np.zeros(Ytest_bin.shape)

#Ntrains is the array of increasing training-sizes
# Ntrains=np.linspace(10,10000,50)
# Ntrains = np.array([int(n) for n in Ntrains])
# zhang_metrics=np.zeros((len(Ntrains),7+s))
#training-size, fbeta, prec, rec, ml-hamm, train-time-sec, test-time-sec

lr_folder = os.listdir('./Clean_Data1Models/')

##################################################
##################################################
##################################################
for lr in list_lr:
    if not ('lr_'+str(lr) in lr_folder):
        os.mkdir('./Clean_Data1Models/lr_'+str(lr))

    #Train and Test for each size in Ntrains    
    for (n_idx, n) in enumerate(Ntrains):
        print("\033[H\033[J")
        print("==============New Iteration==============")
        print("n_idx={}; n={}".format(n_idx,n))
    
        # Configure no. of epochs, learning-rate, and batch-size for training
        epochs = 100
        batchsize = 128
        inp = d
        out = s**2+1
    
        #create tensors for train and validation sets
        Xt = torch.tensor(np.float64(np.load(train_Data_file)))[:n,:]
        Yt = torch.tensor(np.float64(np.load(train_binlabel_file)))[:n,:]
        Xv = torch.tensor(np.float64(np.load(val_Data_file)))[:nval,:]
        Yv = torch.tensor(np.float64(np.load(val_binlabel_file)))[:nval,:]
        #create Train and Validation TensorDatasets
        train_dataset = TensorDataset(Xt,Yt);
        validation_dataset = TensorDataset(Xv,Yv);
    
        # transformation= transforms.Compose([transforms.ToTensor()])
        # train_dataset= Dataset_loader(data_file_name=train_Data_file, label_file_name = train_label_file, transform = transformation)
        # validation_dataset= Dataset_loader(data_file_name=val_Data_file, label_file_name = val_label_file, transform = transformation)
        train_loader= DataLoader(train_dataset,batch_size=batchsize,shuffle = True)
        val_loader= DataLoader(validation_dataset,batch_size=batchsize,shuffle = True)
        
        #Configure the Training Model Architecture
        net = fbeta_dnn1(inp,out);
        net = net.float();
        net.to(device)
        print("Model architecture: ", net)
        
        # Configure the optimizer (loss is Zhang's custom loss)
        # optmizer= optim.SGD(net.parameters(), lr= lr, weight_decay=1e-4)
        optmizer= optim.Adam(net.parameters(), lr= lr, weight_decay=1e-4)
        #weight_decay for l2 regularization
        
        ############################# Start Training Loop ########################
        print("===============Start Training==============")
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
                outputs = net(data.float())
                # Compute Zhang's surrogate loss
                loss = fbeta_loss(outputs,labels)
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
                val_outputs = net(data.float())
                # Compute Zhang's surrogate loss
                loss_ = fbeta_loss(val_outputs, labels)
                #Validation loss
                validation_loss += loss_.item()
                #_, val_preds = torch.max(val_outputs,1)
                #val_correct += torch.sum(val_preds == labels.data)
                ee = ee+1
            
            #compute fbeta score on validation data
            # first generate Yvalpred_bin using decode
            Xv = Xv.to(device)
            Yvalpredscores = net(Xv.float());
            Yvalpredscores = Yvalpredscores.cpu().detach().numpy()
            Yvalpred_bin=np.zeros( (Yvalpredscores.shape[0], s))
            for i in range(Yvalpred_bin.shape[0]):
                Yvalpred_bin[i,:] = decode( Yvalpredscores[i], beta=1)
            #now compute fbeta score
            validation_fbeta = fbeta_score( Yval_bin, Yvalpred_bin, 
                            beta=1, average='samples' )
            
            train_loss = train_loss/len(train_dataset)
            validation_loss = validation_loss/len(validation_dataset)
            #train_acc = train_correct.double() / len(train_dataset)
            print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tValidation Fbeta {:.8f}'
                  .format(e+1, train_loss, validation_loss, validation_fbeta)
                  )
            #Append the training and validation losses/accuracies 
            # to the loss/acc-plot lists
            plot_loss.append(train_loss);
            plot_loss_val.append(validation_loss);
            plot_fbeta_val.append(validation_fbeta)
            if e % 9 == 0 and n_idx==len(Ntrains)-1:
    	    # Save after every certain epochs
                torch.save(
                    net.state_dict(),'./Clean_Data1Models/lr_'+
                    str(lr)+
                    '/zhang-{}-{}-{}-{}-{:.4f}.pt'.format(
                        n,e+1,batchsize,lr,
                        validation_fbeta)
                    )
        traintime_end=time.time()
        train_time_sec=(traintime_end-traintime_start)
        print("=============Training Complete==============")
        #################### End Training Loop #################
        ##################### Start Testing #######################
        print("==============Testing Started=============")
        testtime_start=time.time()
        Xtest = torch.tensor(np.float64(np.load(test_Data_file)))[:ntest,:]
        Xtest = Xtest.to(device)
        Ypredscores = net(Xtest.float())
        Ypredscores = Ypredscores.cpu().detach().numpy()
        ################## Decode the Outputs to Label Predictions################
        print("============Decoding Started==============")
        #Populate Ypred_bin using decode
        for i in range(Ypredscores.shape[0]):
            Ypred_bin[i,:] = decode( Ypredscores[i], beta=1)
        testtime_end=time.time()
        test_time_sec=testtime_end-testtime_start
        print("=============Decoding Complete=============")
        print("============Testing Complete===============")
        ############## Decode the Outputs to Label Predictions############
        #################### End Testing ##########################
        ##################################################
        #We are in (n_idx,n) loop here
        ##################################################
        ###################### Evaluation Metrics#####################
        print("======Metrics Calculation Started==============")
        #compute average precision
        prec=precision_score(Ytest_bin, Ypred_bin, average='micro')        
        #compute average recall
        rec=recall_score(Ytest_bin, Ypred_bin, average='micro') 
        #compute fbeta_score
        fbetascore=fbeta_score(Ytest_bin, Ypred_bin, 
                               beta=1, average='samples')
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
        zhang_metrics[n_idx,:]=np.concatenate(
            ( np.array([n, fbetascore, prec, rec, ml_hamming,
                        train_time_sec, test_time_sec]),
             pl_hamming)
            )
        print("=======Metrics Calculation Complete===========")
        #save the metrics
        # np.save("./../Datasets/Data1/Clean_Metrics/zhang_lr_"+str(lr)+
        #         "_metrics_data1.npy",np.array(zhang_metrics))
        # dictionary={'zhang': zhang_metrics}
        # savemat("./../Datasets/Data1/Clean_Metrics/zhang_lr_"+str(lr)+
        #     "_metrics_data1.mat",
        #     dictionary
        #     )
        #################### Save Excel File ##################
        # columns=['Training-size', 'Fbeta-score', 
        #          'Precision','Recall',
        #          'ML-Hamm','Training-time (sec)', 'Test-time (sec)']
        # for j in range(s):
        #     columns+= ['PL-Hamm-'+str(j)]
        # df = pandas.DataFrame(zhang_metrics, columns=columns)
        # df.to_excel("./../Datasets/Data1/Clean_Metrics/zhang_lr_"+str(lr)+
        #         "_metrics_data1.xlsx" ,index=False)
        ######################### Evaluation Metrics#####################
    ##################################################
    ##################################################
    ##################################################
    ########################### Save Metrics #####################
    # np.save("./../Datasets/Data1/Clean_Metrics/zhang_lr_"+str(lr)+
    #         "_metrics_data1.npy",np.array(zhang_metrics))
    # dictionary={'zhang': zhang_metrics}
    # savemat("./../Datasets/Data1/Clean_Metrics/zhang_lr_"+str(lr)+
    #         "_metrics_data1.mat",
    #         dictionary
    #         )
    #################### Save Excel File ##################
    # columns=['Training-size', 'Fbeta-score', 
    #          'Precision','Recall',
    #          'ML-Hamm','Training-time (sec)', 'Test-time (sec)']
    # for j in range(s):
    #     columns+= ['PL-Hamm-'+str(j)]
    # df = pandas.DataFrame(zhang_metrics, columns=columns)
    # df.to_excel("./../Datasets/Data1/Clean_Metrics/zhang_lr_"+str(lr)+
    #         "_metrics_data1.xlsx" ,index=False)
    ##################################################
    ##################################################
    ##################################################
    ########################### Save Fbeta Accuracy ########################
    # ymdstr = time.strftime("%Y-%m-%d")
    # np.save("./../Datasets/Data1/Metrics/zhang_lr_"+str(lr)+"_fbetaacc_data1_"+ymdstr+".npy",np.array(zhang_metrics))