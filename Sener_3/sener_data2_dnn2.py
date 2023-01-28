#Python Modules
import os
import numpy as np
import time
from tqdm import tqdm
from scipy.io import savemat

#Pandas
import pandas

#Pytorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils import data
import torchvision
from torch.autograd import Variable
from torchvision import transforms

#Our Modules
import losses
from loaders import Dataset_loader
from model_loader2_dnn2 import get_model
from mgda import MGDA_UB

#sk-learn Modules
from sklearn.metrics import fbeta_score, hamming_loss
from sklearn.metrics import precision_score, recall_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set fixed random number seed
torch.manual_seed(42)

#beta=1 by default
# beta=1

# d=72
# s=6

# Define HyperParameters
epochs = 100
batchsize = 128
list_lr = np.logspace(-4,-2,3)
# lr = 0.001

#Tasks/Labels for data2
tasks = ['0','1','2','3','4','5'];

#Load the Dataset files
train_Data_file = './../Datasets/Data2/Xtrain_2021-11-25.npy'
val_Data_file = './../Datasets/Data2/Xval_2021-11-25.npy'
test_Data_file = './../Datasets/Data2/Xtest_2021-11-25.npy'
train_binlabel_file = './../Datasets/Data2/Ytrainbin_2021-11-25.npy'
val_binlabel_file = './../Datasets/Data2/Yvalbin_2021-11-25.npy'
test_binlabel_file = './../Datasets/Data2/Ytestbin_2021-11-25.npy'

#Training Data statistics including d and s
ntrain, d=np.load(train_Data_file).shape
s = np.load(train_binlabel_file).shape[1]

#Validation labels and number of validation points
Yval_bin = np.load(val_binlabel_file);
nval=Yval_bin.shape[0]

#Test labels and number of test points
Ytest_bin = np.load(test_binlabel_file)
ntest=np.load(test_Data_file).shape[0]

#Numpy array for predicted labels
Ypred_bin = np.zeros(Ytest_bin.shape)

#Ntrains is the array of increasing training-sizes
Ntrains=np.linspace(10,312,50)
Ntrains = np.array([int(n) for n in Ntrains])
sener_metrics=np.zeros((len(Ntrains),7+s))
#training-size, fbeta, prec, rec, ml-hamm, train-time-sec, test-time-sec

inp = d
outputs_pertask = 2
dnn_model = 'dnn2';
# dnn1 - inp -> 6 -> out
# dnn2 - inp -> 64 -> 6 -> out
# dnn3 - theta_sh = inp -> 64 -> BN1 -> 16  theta_t = 16 -> 8 -> BN2 -> out

lr_folder = os.listdir('./Data2Models/')

for lr in list_lr:
    if not ('lr_'+str(lr) in lr_folder):
        os.mkdir('./Data2Models/lr_'+str(lr))
    
    #Train and Test for each size in Ntrains
    for (n_idx, n) in enumerate(Ntrains):
        print("\033[H\033[J")
        print("==============New Iteration==============")
        print("n_idx={}; n={}".format(n_idx,n))
        
        transformation = transforms.Compose([transforms.ToTensor()])
        
        train_dataset = Dataset_loader(data_file_name = train_Data_file,
                                       label_file_name = train_binlabel_file, 
                                       transform = transformation,n=n)
        val_dataset = Dataset_loader(data_file_name = val_Data_file, 
                                     label_file_name = val_binlabel_file, 
                                     transform = transformation,n=n)
        
        train_loader= DataLoader(train_dataset, batch_size=batchsize, 
                                 shuffle = True)
        val_loader= DataLoader(val_dataset, batch_size=batchsize, 
                               shuffle = True)
        
        model = get_model(parallel=False, tasks=tasks, 
                          inp=inp,
                          out=outputs_pertask);
        model_params = []
        for m in model:
            model_params += model[m].parameters()
        
        # Define loss and optimizer
        nll_loss = losses.get_loss(tasks)
        optimizer = torch.optim.Adam(model_params, lr=lr, weight_decay=1e-4)
        
        # Evaluation metric to use during training/validation
        def get_accuracy(labels, ypred, hamacc_t, fbetaacc_t, t):
            ypred = np.argmax(ypred.cpu().detach().numpy(), 1)
            fbetaacc_t[t].append( fbeta_score(labels[t], ypred, beta=1) )
            hamacc_t[t].append( np.sum(ypred==labels[t].detach().numpy()) )
            return hamacc_t, fbetaacc_t
        
        ############################# Start Training ########################
        print("============Start Training=============")
        traintime_start=time.time()
        n_iter = 0
        loss_init = {}
        hammacc_train, hammacc_val = {},{}
        # for epoch in tqdm(range(epochs)):
        for epoch in range(epochs):
            fbeta_train, fbeta_val = {},{}
            for t in tasks:
                hammacc_train[t], fbeta_train[t]=[],[]
                hammacc_val[t], fbeta_val[t]=[],[]
            print('\nEpoch {} Started'.format(epoch))
        #    if (epoch+1) % 10 == 0:
        #        # Every 50 epoch, half the lr
        #        for param_group in optimizer.param_groups:
        #            param_group['lr'] *= 0.85
        #        print('Half the learning rate{}'.format(n_iter))

            for m in model:
                model[m].train()
        
            for batch in train_loader:
                n_iter += 1
                # print(n_iter)
                X_train_batch = batch[0]
                labels = {}
                for i, t in enumerate(tasks):
                    labels[t] = batch[i+1]
                # Scaling the loss functions based on the algorithm choice
                loss_data = {}
                grads = {}
                scale = {}
                # MGDA-UB loop
                optimizer.zero_grad()
                X_train_batch = X_train_batch.data;
                rep = model['rep'](X_train_batch.float())           
                # Gradient wrt z (latent variable)
                rep_variable = Variable(rep.data.clone(), 
                                     requires_grad=True)
                for t in tasks:
                    optimizer.zero_grad()
                    y_pred = model[t](rep_variable)
                    loss = nll_loss[t](y_pred, labels[t])
                    loss_data[t] = loss.data
                    loss.backward()
                    grads[t] = []
                    grads[t].append(Variable(rep_variable.grad.data.clone(), 
                                       requires_grad=False))
                    rep_variable.grad.data.zero_()
            # FRANK-WOLFE Solver
                sol, min_norm = MGDA_UB.FW_solver([grads[t][0] for t in tasks])
                for i, t in enumerate(tasks):
                    scale[t] = float(sol[i])
                # Scaled back-propagation
                optimizer.zero_grad()
                rep = model['rep'](X_train_batch.float())
                for i, t in enumerate(tasks):
                    y_pred_t = model[t](rep)
                    loss_t = nll_loss[t](y_pred_t, labels[t])
                    hammacc_train, fbeta_train = get_accuracy(labels,y_pred_t,
                                                 hammacc_train, fbeta_train,
                                                 t)
                    loss_data[t] = loss_t.data
                    if i > 0:
                        loss = loss + scale[t]*loss_t
                    else:
                        loss = scale[t]*loss_t
                loss.backward()
                optimizer.step()
        
            #Validate the model in each epoch
            for m in model:
                model[m].eval()
            tot_loss = {}
            tot_loss['all'] = 0.0
            for t in tasks:
                tot_loss[t] = 0.0
            num_val_batches = 0
            for batch_val in val_loader:
                X_val_batch = batch_val[0]
                labels_val = {}
                for i, t in enumerate(tasks):
                    labels_val[t] = batch_val[i+1]
                val_rep= model['rep'](X_val_batch.float())
                for t in tasks:
                    y_pred_t_val = model[t](val_rep)
                    loss_t = nll_loss[t](y_pred_t_val, labels_val[t])
                    tot_loss['all'] += loss_t.data
                    tot_loss[t] += loss_t.data
                    hammacc_val, fbeta_val = get_accuracy(labels_val,y_pred_t_val,
                                                  hammacc_val,fbeta_val,
                                                  t)
                num_val_batches+=1
            mean_fbeta_val = np.mean([fbeta_val[k] for k in fbeta_val.keys()])
            mean_fbeta_train = np.mean([fbeta_train[k] for k in fbeta_train.keys()])
            print("Mean Val F-beta score: ", mean_fbeta_val)
            print("Mean Train F-beta score: ", mean_fbeta_train)
            
            if epoch % 9 == 0 and n_idx==len(Ntrains)-1:
                # Save after every certain epoch
                state = {'epoch': epoch+1,
                        'model_rep': model['rep'].state_dict(),
                        'optimizer_state' : optimizer.state_dict()}
                for t in tasks:
                    key_name = 'model_{}'.format(t)
                    state[key_name] = model[t].state_dict()
                torch.save(
                    state,'./Data2Models/lr_'+str(lr)+'/sener_'+dnn_model+
                    '-{}-{}-{}-{}-{:.4f}.pt'.format(
                        n,epoch+1,batchsize,lr,
                        mean_fbeta_val)
                    )
        traintime_end=time.time()
        train_time_sec=(traintime_end-traintime_start)
        print("===============Training Complete==============")
        ############################# End Training ########################
        ############################# Start Testing ############################
        #Perform Testing using the Trained Model
        print("============Testing Started================")
        testtime_start=time.time()
        test_dataset = Dataset_loader(data_file_name = test_Data_file,
                                      label_file_name = test_binlabel_file, 
                                      transform = transformation,n=ntest)
        
        test_loader= DataLoader(test_dataset, batch_size=ntest,
                                shuffle = True)
        #########Generate Predicted labels and populate Ypred_bin#############
        for batch_test in test_loader:
            val_images = batch_test[0];
            labels_val={}
        
            for i, t in enumerate(tasks):
                if t not in tasks:
                    continue
                labels_val[t] = batch_test[i+1]
        
            val_rep = model['rep'](val_images.float())
            for t in tasks:
                out_t_val = model[t](val_rep)
                ypred = np.argmax(out_t_val.detach().numpy(),axis=1);
                Ypred_bin[:,int(t)] = ypred
        testtime_end=time.time()
        test_time_sec=testtime_end-testtime_start
        print("=============Testing Complete===============")
        #########Generate Predicted labels and populate Ypred_bin#############
        ############################# End Testing ############################
        ##################################################
        #We are in (n_idx,n) loop here
        ##################################################
        ########################### Evaluation Metrics########################
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
        sener_metrics[n_idx,:]=np.concatenate(
            ( np.array([n, fbetascore, prec, rec, ml_hamming,
                        train_time_sec, test_time_sec]),
             pl_hamming)
            )
        print("=======Metrics Calculation Complete===========")
        #save the metrics
        np.save("./../Datasets/Data2/Metrics/sener_"+dnn_model+"_lr_"+str(lr)+
                "_metrics_data2.npy",np.array(sener_metrics))
        dictionary={'sener': sener_metrics}
        savemat("./../Datasets/Data2/Metrics/sener_"+dnn_model+"_lr_"+str(lr)+
            "_metrics_data2.mat",
            dictionary
            )
        ########################### Evaluation Metrics########################
    ##################################################
    ##################################################
    ##################################################
    ########################### Save Metrics #####################
    np.save("./../Datasets/Data2/Metrics/sener_"+dnn_model+"_lr_"+str(lr)+
            "_metrics_data2.npy",np.array(sener_metrics))
    dictionary={'sener': sener_metrics}
    savemat("./../Datasets/Data2/Metrics/sener_"+dnn_model+"_lr_"+str(lr)+
        "_metrics_data2.mat",
        dictionary
        )
    #################### Save Excel File ##################
    columns=['Training-size', 'Fbeta-score', 
             'Precision','Recall',
             'ML-Hamm','Training-time (sec)', 'Test-time (sec)']
    for j in range(s):
        columns+= ['PL-Hamm-'+str(j)]
    df = pandas.DataFrame(sener_metrics, columns=columns)
    df.to_excel("./../Datasets/Data2/Metrics/sener_"+dnn_model+"_lr_"+str(lr)+
        "_metrics_data2.xlsx" ,index=False)