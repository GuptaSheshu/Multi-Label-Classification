# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 23:04:56 2021

@author: khann
"""
import numpy as np

#define gamma which is the logit link function
def gamma(p: np.array):
    return np.log( p/(1-p) )

#define gamma_inverse
def gammainv(t: np.array):
    return np.exp(t)/( 1+np.exp(t) )
    
# def decode(u0: float, u: dict, beta=1):
def decode(fs: np.array, beta=1):
    s=int( np.sqrt( len(fs)-1 ) )
    u0=fs[0]
    U=np.zeros((s,s))
    ct=1
    for j in range(s):
        for k in range(s):
            U[j,k]=fs[ct]
            ct+=1
    #compute matrix Q
    Q=gammainv( U )
    #compute matrix V
    V=np.zeros((s,s))
    for k in range(s):
        for l in range(s):
            V[k,l]=-(1+beta)**2 / ( (k+1)*beta**2+ (l+1) )
    #compute matrix T=QV
    T=Q@V
    
    Y = np.zeros( (s,s) )
    z = {}
    for l in range(s):
        #find l+1 smallest values in T[:,l]
        temp=np.argsort(T[:,l])
        temp=temp[:l+1]
        candidates=T[temp,l]
        #configure Y[l]
        for j in range(s):
            if j in temp:
                Y[l][j]=1
            else:
                Y[l][j]=0
        z[l+1]=np.sum(candidates)
    
    #compute scores
    f={}
    f[0]=-gammainv(u0)
    for k in range(1,s+1):
        f[k]=z[k]
    # print(f)
    
    #find minimum
    no_of_ones=int( min(f, key=f.get))
    if no_of_ones==0:
        return np.zeros(s)
    else:
        return Y[no_of_ones-1,:]
    