# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 19:53:17 2020

@author: Jing
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math

npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']

# face1 = trainface[2400,:].reshape((19,19))
# plt.imshow(face1,cmap='gray')

def BPNNtrain(pf,nf,hn,lr,iteration):
    #positive_feature: 正資料 (2429,361)
    #negative_feature: 負資料 (4548,361)
    #hidden_nod,learning_rate,iteration
    pn = pf.shape[0] #2429
    nn = nf.shape[0] #4548
    fn = pf.shape[1] #361
    feature = np.append(pf,nf,axis=0)
    target = np.append(np.ones((pn,1)),np.zeros((nn,1)),axis=0)
    WI = np.random.normal(0,1,(fn+1,hn))
    WO = np.random.normal(0,1,(hn+1,1))
    for t in range(iteration):
        s = random.sample(range(pn+nn),pn+nn) #產生亂數序
        for i in range(pn+nn):
            ins = np.append(feature[s[i],:],1) #input signal
            ho = ins.dot(WI) #hidden node output <-矩陣相乘
            for j in range(hn):
                ho[j] = 1/(1+math.exp(-ho[j])) #hidden output轉成sigmoid函數
            hs = np.append(ho,1)
            out = hs.dot(WO) #output <-矩陣相乘
            out = 1/(1+math.exp(-out)) #output也轉sigmoid
            dk = out*(1-out)*(target[s[i]]-out) #delta k
            dh = ho*(1-ho)*WO[:hn,0]*dk #只取前20個hidden node(第21個是常數項不要)去算delta h
                                        #也可用for迴圈去算
            WO[:,0] += lr*dk*hs #21個值一起更新        
            for j in range(hn):
                WI[:,j] += lr*dh[j]*ins                                
    return WI,WO

def BPNNtest(feature,WI,WO):
    #feature: 未知的資料
    #WI,WO: 目前學到的模型
    sn = feature.shape[0] #sanple number
    hn = WI.shape[1] 
    out = np.zeros((sn,1))
    for i in range(sn):
        ins = np.append(feature[i,:],1)
        ho = ins.dot(WI) #hidden node output <-矩陣相乘
        for j in range(hn):
            ho[j] = 1/(1+math.exp(-ho[j])) #hidden output轉成sigmoid函數
        hs = np.append(ho,1)
        out[i] = hs.dot(WO) #output <-矩陣相乘
        out[i] = 1/(1+math.exp(-out[i])) #output也轉sigmoid
    return out

WI, WO = BPNNtrain(trainface/255,trainnonface/255,20,0.01,10) #正規化到01之間
pscore = BPNNtest(trainface/255,WI,WO)
nscore = BPNNtest(trainnonface/255,WI,WO)

#要讓準確度提高可增加hn,iter
#根據pscore nscore的分數可以去看那些照片學不好
#目標是讓pscore接近1，nscore接近0

pn = np.shape(pscore)[0] # num of positive sample
nn = np.shape(nscore)[0]
FPR = np.zeros((1001,1))
TPR = np.zeros((1001,1))
for i in range(1001):
    threshold = i/1000
    FPR[i] = np.sum(nscore>=threshold)/nn
    TPR[i] = np.sum(pscore>=threshold)/pn
plt.plot(FPR,TPR)