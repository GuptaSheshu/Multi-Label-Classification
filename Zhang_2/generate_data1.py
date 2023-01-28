# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 12:00:52 2021

@author: khann
"""
import numpy as np
import random
import time
import matplotlib.pyplot as plt

#random seed
random.seed(0)

####################FUNCTION DEFINITIONS#####################
#############################################################
def convert2bin(a: int, s: int):
    temp=bin(a).replace("0b","").zfill(s)
    return np.array([int(c) for c in temp])

def get_a(y: int, j: int, k: int, s=6):
    #j and k denote indices wrt Fbeta paper
    #convert y to its binary representation
    # ystring=bin(y).replace("0b","").zfill(s)
    ybin = convert2bin(y,s)
    sum_row = np.sum(ybin)   
    temp1 = int((sum_row==k))
    temp2 = int( (ybin[j-1]==1) )
    if j==0:
        return temp1;
    else:
        return temp1*temp2

#define gamma which is the logit link function
def gamma(q: np.array):
    # np.where(q==1, print(np.where(q==1)), print(""))
    # for i in range(len(p)):
    #     if p[i]==1:
    #         print(i)
    return np.log( q/(1-q) )

#define gamma_inverse
def gammainv(t: np.array):
    return np.exp(t)/( 1+np.exp(t) )
####################FUNCTION DEFINITIONS#####################
#############################################################

#choose any d>0 and s>0 (s should not be too large for test)
d=20
s=4
beta=1
ntrain=10000
nval = 2000
ntest=15000
n = ntrain+nval+ntest

#generate a fixed W matrix
W=np.zeros( (int(s**2+1), int(d)) )

#make sure W is full-rank
while np.linalg.matrix_rank(W)<min(int(s**2+1), int(d) ):
    W = np.random.rand(int(s**2+1), int(d))

#generate a fixed vector alpha
alpha=np.zeros(2**s)
for i in range(len(alpha)):
    #generate each entry independently from Unif([0.1,1])
    alpha[i]=np.random.uniform(0.1,1)

#now generate training data
X=np.zeros((n, d))
Y=np.zeros(n)
Ybin=np.zeros((n, s))
# Ptest=np.zeros((ntest, 2**s))
P=np.random.dirichlet(alpha, n)

for i in range(n):
    #generate probability simplex on an alphabet of size 2^s
    p=P[i,:]    
    q=np.zeros((s**2+1))
    #compute vector q
    #first form matrix A and then multiply with p
    A = np.zeros((s**2+1,2**s))
    for y in range(2**s):
        A[0][y] = get_a(y,0,0,s)
    for y in range(2**s):
        ct=1
        for j in range(1,s+1):
            for k in range(1,s+1):
                A[ct,y]=get_a(y,j,k,s)
                ct+=1
    q=A@p
                  
    # q={}
    # q[0]=p[0]
    # for j in range(1,s+1):
    #     for k in range(1,s+1):
    #         q[j][k]=0
    # 
    # for i in range(1, len(alpha)):
    #     temp = bin(i).replace("0b","").zfill(s)
    #     temp = np.array([int(c) for c in temp])
    #     k=int ( np.sum(temp) )
    #     # if k>0:
    #     for j in range(s):
    #         if temp[j]==1:
    #             q[j+1,k]+=p[i]
        # else:
            #k should not be > 0
            # print("Some bug")
    
    #generate x
    x=np.linalg.pinv(W)@gamma(q)
    #generate y
    y=np.random.choice(range(2**s),p=p)
    #store x,y, and ybin
    X[i,:]=x
    Y[i]=y
    Ybin[i,:]=convert2bin(y,s)
    # ystring=bin(y).replace("0b","").zfill(s)
    # ybin = np.array([int(c) for c in ystring])
    # Ytestbin[i,:]=ybin
    # Ptest[i,:]=p

#Save Xtest, Ytest, Ptest in numpy files
timestr = time.strftime("%Y-%m-%d")
#Save Train Files
np.save("Data1/Xtrain_"+timestr+".npy",X[0:ntrain,:])
np.save("Data1/Ytrain_"+timestr+".npy",Y[0:ntrain])
np.save("Data1/Ytrainbin_"+timestr+".npy",Ybin[0:ntrain,:])
np.save("Data1/Ptrain_"+timestr+".npy",P[0:ntrain,:])
#Save Validation Files
np.save("Data1/Xval_"+timestr+".npy",X[ntrain:ntrain+nval,:])
np.save("Data1/Yval_"+timestr+".npy",Y[ntrain:ntrain+nval])
np.save("Data1/Yvalbin_"+timestr+".npy",Ybin[ntrain:ntrain+nval,:])
np.save("Data1/Pval_"+timestr+".npy",P[ntrain:ntrain+nval,:])
#Save Test Files
np.save("Data1/Xtest_"+timestr+".npy",X[ntrain+nval:n,:])
np.save("Data1/Ytest_"+timestr+".npy",Y[ntrain+nval:n])
np.save("Data1/Ytestbin_"+timestr+".npy",Ybin[ntrain+nval:n,:])
np.save("Data1/Ptest_"+timestr+".npy",P[ntrain+nval:n,:])

#compute bayes optimal classifier's Fbeta accuracy
Fbeta=np.zeros((2**s,2**s))
for yhat in range(2**s): #yhats are along the rows
    for yt in range(2**s): #ys are along the columns
        if yhat==0 and yt==0:
            Fbeta[yhat][yt]=1
        else:
            temp=bin(yhat).replace("0b","").zfill(s)
            yhatbin=np.array([int(c) for c in temp])
            temp=bin(yt).replace("0b","").zfill(s)
            ytbin=np.array([int(c) for c in temp]) 
            Fbeta[yhat][yt]=(1+beta**2)*np.dot(ytbin,yhatbin)/(
                (beta**2)*np.sum(ytbin)+np.sum(yhatbin)
                )

Ptest = np.load("Data1/Ptest_"+timestr+".npy")
E=Fbeta@(Ptest.transpose())
Ypred=np.argmax(E,0)
Ypredbin=np.zeros((ntest,s))
for i in range(len(Ypred)):
    temp=bin(Ypred[i]).replace("0b","").zfill(s)
    Ypredbin[i,:]=np.array([int(c) for c in temp])
# Ypredbin=np.vectorize(convert2bin)(Ypred,s)

#compute accuracy of Bayes optimal classifier
bayes_acc=0
# Ytest = np.load("Data1/Ytest_"+timestr+".npy")
Ytestbin = np.load("Data1/Ytestbin_"+timestr+".npy")

for i in range(ntest):
    ytestbin=Ytestbin[i,:]
    ypredbin=Ypredbin[i,:]
    temp=(1+beta**2)*np.dot(ytestbin,ypredbin)/(
        (beta**2)*np.sum(ytestbin)+np.sum(ypredbin)
        )
    bayes_acc+=(1/ntest)*temp
print(bayes_acc)
np.save("Data1/bayes_fbetaacc_data1_"+timestr+".npy",np.array([bayes_acc]))

#Plot the horizontal for accuracy
# Create figure
# Ntrains=[10, 100, 500, 1000, 1500, 5000, 7500, 10000, 15000, 20000]
# zhang_acc=[0.63]*len(Ntrains)
# fig, (ax1) = plt.subplots(1,1)
# #first plot bayes optimal performance
# ax1.semilogx(Ntrains,[bayes_acc]*len(Ntrains)  )
# ax1.semilogx(Ntrains,zhang_acc)
# ax1.set(title="Correctness of Zhang's Algorithm's Implementation")
# ax1.grid()
# ax1.set_ylim(top=0.7)
# ax1.set_ylim(bottom=0.56)
# fig.tight_layout()
# plt.legend(["Bayes Optimal", "Zhang's Algorithm"])
# plt.show()

# for ntrain in Ntrain[10, 100, 200, 400, 800, 20000]
