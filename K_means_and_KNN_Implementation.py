#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import random
import collections
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import datasets


# In[2]:


#Q1
iris = datasets.load_iris()
iris.data


# In[3]:


iris.target


# In[4]:


#(1)未經標準化
origin_irisdata = iris.data
origin_irisdata


# In[5]:


iris = datasets.load_iris()

#mean
mean = []
for i in range(iris.data.shape[1]):
    total = 0
    for j in range(iris.data.shape[0]):
        total += iris.data[j,i]
    mean.append(total/iris.data.shape[0])
print(mean)

#standard deviation
std = []
for i in range(iris.data.shape[1]):
    result = 0
    for j in range(iris.data.shape[0]):
        result += (iris.data[j,i]-mean[i])**2
    std.append(math.sqrt(result/iris.data.shape[0]))
print(std)        


# In[6]:


#(2)Standard Score Normalize
for i in range(iris.data.shape[1]):
    for j in range(iris.data.shape[0]):
        iris.data[j,i] = (iris.data[j,i]-mean[i])/std[i]

StandardScore_irisdata = iris.data
StandardScore_irisdata


# In[7]:


iris = datasets.load_iris()

#min,max
min = []
max = []
for i in range(iris.data.shape[1]):
    min_temp = 100
    max_temp = 0
    for j in range(iris.data.shape[0]):
        if iris.data[j,i] < min_temp:
            min_temp = iris.data[j,i]
        if iris.data[j,i] > max_temp:
            max_temp = iris.data[j,i]
    min.append(min_temp)
    max.append(max_temp)
print(min)
print(max)


# In[8]:


#(3)Scaling Normalize
for i in range(iris.data.shape[1]):
    for j in range(iris.data.shape[0]):
         iris.data[j,i] = (iris.data[j,i]-min[i])/(max[i]-min[i])

Scaling_irisdata = iris.data
Scaling_irisdata


# In[9]:


#kmeans
def kmeans(sample,K,maxiter):

    N = sample.shape[0] #sample筆數
    D = sample.shape[1] #維度
    C = np.zeros((K,D)) 
    L = np.zeros((N,1)) #N個sample的label
    L1 = np.zeros((N,1)) #存上一次的label
    dist = np.zeros((N,K)) 
    idx = random.sample(range(N),K) #在range(N)中任取K個當center
    C = sample[idx,:] #取center的座標
    iter = 0
    while iter < maxiter:
        for i in range(K):
            dist[:,i] = np.sum((sample-np.tile(C[i,:],(N,1)))**2,1) 
            #每個sample算與N個點的距離
            #np.tile(---,(N,1)) -> 往下copy N 次，往右copy 1 次
            #sum(---,1) -> 算距離平方；1代表往右加
        L = np.argmin(dist,1) #取最小距離的那個點(0 or 1 or 2)；1代表往右算
        if np.array_equal(L,L1): #這次最小距離的點跟上次一樣
            break
        L1 = L
        for i in range(K):
            idx = np.nonzero(L==i)[0] #取出某一群的所有點的index
            if len(idx)>0: #這群有點
                C[i,:] = np.mean(sample[idx,:],0) #取中心點
        iter += 1
        #print(iter)
    return C,L


# In[10]:


#未標準化之分群結果
ori = kmeans(origin_irisdata,3,1000)[1]
print(ori)


# In[11]:


#Standard Score之分群結果
std = kmeans(StandardScore_irisdata,3,1000)[1]
print(std)


# In[12]:


#Scaling之分群結果
sca = kmeans(Scaling_irisdata,3,1000)[1]
print(sca)


# In[13]:


#Evaluate accuracies under three conditions

#看資料範圍內出現最多次的label是0 or 1 or 2 
def ClusteredTeam(data,start,end):
    count=[0,0,0] #每一群各label的記數
    for i in range(start,end):
        if data[i]==0:
            count[0]+=1
        elif data[i]==1:
            count[1]+=1
        else:
            count[2]+=1
    #print('cluster result:',count)
    return np.argmax(count)

#算分群準確率
def Accuracy(data):
    #每50筆資料(同一群)去看出現最多次的label
    t1 = ClusteredTeam(data,0,50)
    t2 = ClusteredTeam(data,51,100)
    t3 = ClusteredTeam(data,101,150)

    error = 0 #分類錯誤數
    for i in range(len(data)):
        #第一群
        if i<50:
            if data[i] != t1:
                error += 1
        #第二群
        elif i<100:
            if data[i] != t2:
                error +=1
        #第三群
        else:
            if data[i] != t3:
                error += 1
    return (150-error)/150

print('未標準化之Accuracy:',Accuracy(ori))
print('Standard score之Accuracy:',Accuracy(std))
print('Scaling之Accuracy:',Accuracy(sca))


# In[14]:


#Q2
iris = datasets.load_iris()
X = iris.data
Y = iris.target


# In[15]:


def knn(test,train,target,K):
    N = train.shape[0]
    dist = np.sum((train-np.tile(test,(N,1)))**2,1)
    idx = sorted(range(len(dist)) , key=lambda i:dist[i])[0:K] #排序0-148，取前K個(排序的依據是dist[i])
    return collections.Counter(target[idx]).most_common(1)[0][0]


# In[16]:


#用3*3零矩陣存混淆矩陣結果
CM_1 = np.zeros((3,3))
CM_10 = np.zeros((3,3))
#左:預測；上:真實

for i in range(len(X)):
    X1 = np.concatenate((X[:i],X[(i+1):])) # x train
    Y1 = np.concatenate((Y[:i],Y[(i+1):])) # y train
    X2 = X[i] # x test
    Y2 = Y[i] # y test
    #1-NN        
    result_1 = knn(X2,X1,Y1,1)
    #分群結果為0
    if result_1==0:
        #分對
        if result_1==Y[i]:
            CM_1[0,0]+=1
        #分錯
        else:
            if Y[i]==1:
                CM_1[0,1]+=1
            elif Y[i]==2:
                CM_1[0,2]+=1
    #分群結果為1
    elif result_1==1:
        #分對
        if result_1==Y[i]:
            CM_1[1,1]+=1
        #分錯
        else:
            if Y[i]==0:
                CM_1[1,0]+=1
            elif Y[i]==2:
                CM_1[1,2]+=1
    #分群結果為2
    elif result_1==2:
        #分對
        if result_1==Y[i]:
            CM_1[2,2]+=1
        #分錯
        else:
            if Y[i]==0:
                CM_1[2,0]+=1
            elif Y[i]==1:
                CM_1[2,1]+=1
    #10-NN
    result_10 = knn(X2,X1,Y1,10)
    if result_10==0:
        if result_10==Y[i]:
            CM_10[0,0]+=1
        else:
            if Y[i]==1:
                CM_10[0,1]+=1
            elif Y[i]==2:
                CM_10[0,2]+=1
    elif result_10==1:
        if result_10==Y[i]:
            CM_10[1,1]+=1
        else:
            if Y[i]==0:
                CM_10[1,0]+=1
            elif Y[i]==2:
                CM_10[1,2]+=1
    elif result_10==2:
        if result_10==Y[i]:
            CM_10[2,2]+=1
        else:
            if Y[i]==0:
                CM_10[2,0]+=1
            elif Y[i]==1:
                CM_10[2,1]+=1
                
print('1-NN:\n',CM_1)
print('10-NN:\n',CM_10)


# In[ ]:




