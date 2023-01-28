# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:38:39 2021

@author: khann
"""
import numpy as np
Xtrain=np.load('Xtrain_2021-11-24.npy')
Xtest=np.load('Xtest_2021-11-24.npy')
Ytrain=np.load('Ytrain_2021-11-24.npy')
Ytrain_bin=np.load('Ytrainbin_2021-11-24.npy')
Ytest=np.load('Ytest_2021-11-24.npy')
Ytest_bin=np.load('Ytestbin_2021-11-24.npy')

n,d=Xtrain.shape
cutoff=int(n*(4/5))
Xtrain2=Xtrain[:cutoff,:]
Ytrain2=Ytrain[:cutoff]
Ytrain2_bin=Ytrain_bin[:cutoff,:]

Xval=Xtrain[cutoff:,:]
Yval=Ytrain[cutoff:]
Yval_bin=Ytrain_bin[cutoff:,:]

np.save('./Data2/Xtrain_2021-11-25.npy',Xtrain2)
np.save('./Data2/Ytrain_2021-11-25.npy',Ytrain2)
np.save('./Data2/Ytrainbin_2021-11-25.npy',Ytrain2_bin)

np.save('./Data2/Xval_2021-11-25.npy',Xval)
np.save('./Data2/Yval_2021-11-25.npy',Yval)
np.save('./Data2/Yvalbin_2021-11-25.npy',Yval_bin)