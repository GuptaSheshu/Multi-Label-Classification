import numpy as np
from scipy.io import arff
import pandas as pd

#Visualizing few data points

data = arff.loadarff('/Users/upasanathakuria/Desktop/emotions/emotions-train.arff')
df = pd.DataFrame(data[0])

df.head()

##For Training
data, meta = arff.loadarff('./emotions-train.arff') #72 numeric, 6 labels (593x78)
#print(data) 
#data[390][77] #391x78 #no. of data point=391

d=72 #dimension of feature vector
s=6 #no. of labels
n=391 #no. of training data points

x_train=[]
y_train=[]
x_t = np.zeros((n, 1))
y_t = np.zeros((n, 1))

for i in range(n):
    for j in range(d):
        x_t=data[i][j]
        #print(x_t)
        x_train.append(x_t)
        
for i in range(n):
    for j in range(d,d+s):
        y_t=data[i][j]
        #print(y_t)
        y_train.append(y_t)
x_train=np.array(x_train)
x_train=np.reshape(x_train, (n, d))
#x_train.shape #(391, 72)
y_train=np.array(y_train, dtype='int64')
y_train=np.reshape(y_train, (n, s))


np.save('Xtrain_2021-11-24.npy', x_train)
np.save('Ytrainbin_2021-11-24.npy', y_train)
np.load("Ytrainbin_2021-11-24.npy")

#Converting y_train into decimal form
def bool2int(x):
    y = 0.0
    for i,j in enumerate(x):
        y += j<<i
    return y

y_train_decimal = [bool2int(x[::-1]) for x in y_train]
y_train_decimal = np.array(y_train_decimal)
#y_train_decimal

np.save('Ytrain_2021-11-24.npy', y_train_decimal)
np.load('Ytrain_2021-11-24.npy')

##For Testing
ntest=202 #no. of training data points

data2, meta = arff.loadarff('./emotions-test.arff') #72 numeric, 6 labels (202x78)

x_test=[]
y_test=[]
x_te = np.zeros((ntest, 1))
y_te = np.zeros((ntest, 1))


for i in range(ntest):
    for j in range(d):
        x_te=data2[i][j]
        #print(x_t)
        x_test.append(x_te)
        
for i in range(ntest):
    for j in range(d,d+s):
        y_te=data2[i][j]
        #print(y_t)
        y_test.append(y_te)
x_test=np.array(x_test)
x_test=np.reshape(x_test, (ntest, d))
x_test
y_test=np.array(y_test, dtype='int64')
y_test=np.reshape(y_test, (ntest, s))
y_test.shape #(202, 6)

np.save('Xtest_2021-11-24.npy', x_test)
np.save('Ytestbin_2021-11-24.npy', y_test)

#Converting y_test into decimal form
def bool2int(x):
    y = 0.0
    for i,j in enumerate(x):
        y += j<<i
    return y

y_test_decimal = [bool2int(x[::-1]) for x in y_test]
y_test_decimal = np.array(y_test_decimal)
#y_train_decimal

np.save('Ytest_2021-11-24.npy', y_test_decimal)
np.load('Ytest_2021-11-24.npy')
#y_test_decimal.shape


##Using 1/4th of the total training data for validation

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

np.save('Xtrain_2021-11-25.npy',Xtrain2)
np.save('Ytrain_2021-11-25.npy',Ytrain2)
np.save('Ytrainbin_2021-11-25.npy',Ytrain2_bin)

np.save('Xval_2021-11-25.npy',Xval)
np.save('Yval_2021-11-25.npy',Yval)
np.save('Yvalbin_2021-11-25.npy',Yval_bin)