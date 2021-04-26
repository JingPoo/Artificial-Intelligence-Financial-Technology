#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np


# In[2]:


def entropy(p1,n1):
    if p1==0 and n1==0:
        return 1
    elif p1==0 or n1 ==0:
        return 0
    else:
        pp = p1/(p1+n1)
        np = n1/(p1+n1)
        return -pp*math.log2(pp)-np*math.log2(np)
    
#information gain  
def IG(p1,n1,p2,n2):
    num = p1+n1+p2+n2
    num1 = p1+n1
    num2 = p2+n2
    return entropy(p1+p2,n1+n2)-num1/num*entropy(p1,n1)-num2/num*entropy(p2,n2)


# In[3]:


#load iris dataset
from sklearn import datasets
iris = datasets.load_iris()
iris


# In[4]:


#刪除target=2的花
iris.target = iris.target[:100]
iris.data = iris.data[:100]
#將array打亂
np.random.seed(1)
np.random.shuffle(iris.target)
np.random.seed(1)
np.random.shuffle(iris.data)
iris


# In[5]:


#建樹function
def buildTree(data,target):
    node = dict() 
    node['data']= range(len(target)) 
    Tree = []
    Tree.append(node) #把node丟進樹
    t = 0
    while(t<len(Tree)):
        idx = Tree[t]['data'] #這個node的所有data
        if sum(target[idx])==0: #如果落在此node的sample都是0
            Tree[t]['leaf'] = 1 #他就當作葉節點
            Tree[t]['decision'] = 0 
        elif sum(target[idx]==len(idx)): #如果落在此node的sample都是1
            Tree[t]['leaf'] = 1
            Tree[t]['decision'] = 1 
        else:
            bestIG = 0
            for i in range(data.shape[1]): #shape[1] -> feature有幾個col 
                pool = list(set(data[idx,i])) #set -> 集合(元素不重複)，第i個feature出現的數字放進set
                pool.sort()
                for j in range(len(pool)-1): #從左到右開始切所有可能 ex. 1|23 , 12|3
                    thres = (pool[j]+pool[j+1])/2 
                    G1 = [] #group1
                    G2 = [] #group2
                    for k in idx: #在這個node的所有數字中
                        if iris.data[k,i]<=thres: #用上面切的刀分group
                            G1.append(k)
                        else:
                            G2.append(k)
                    p1 = sum(target[G1]==1)
                    n1 = sum(target[G1]==0)
                    p2 = sum(target[G2]==1)
                    n2 = sum(target[G2]==0)
                    thisIG = IG(p1,n1,p2,n2)
                    if thisIG>bestIG:
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres #紀錄這次的切法
                        bestf = i #紀錄這次選哪個特徵去切那一刀
            if bestIG>0: #如果這一刀切的有意義
                Tree[t]['leaf'] = 0
                Tree[t]['selectf'] = bestf
                Tree[t]['threshold'] = bestthres
                Tree[t]['child'] = [len(Tree),len(Tree)+1]
                node = dict()
                node['data'] = bestG1
                Tree.append(node)
                node = dict()
                node['data'] = bestG2
                Tree.append(node)
            else: #最好的切法都沒有information
                Tree[t]['leaf'] = 1
                if sum(target[idx]==1)>sum(target[idx]==0):
                    Tree[t]['decision'] = 1
                else:
                    Tree[t]['decision'] = 0
        t+=1
    return Tree


# In[6]:


#測試function
def test(data,target,Tree):
    error = 0
    for i in range(len(target)): #把測試資料丟進去測試是否分對
        test_feature = data[i,:]
        now = 0 #從root開始
        while Tree[now]['leaf']==0:
            bestf = Tree[now]['selectf']
            thres = Tree[now]['threshold']
            if test_feature[bestf]<=thres:
                now = Tree[now]['child'][0]
            else:
                now = Tree[now]['child'][1]
        print(target[i],Tree[now]['decision'])
        if target[i]!=Tree[now]['decision']:
            error += 1
    if len(target)==0:
        return 0
    else:
        return error/len(target)


# In[7]:


#k-fold cross validation
def K_fold_CV(k,data):
    error = 0
    #設定subset size 即data長度/k
    subset_size = int(len(data)/k)
    for i in range(k):
        #每次取出1群測試，其他9群建模，重覆10次
        X_test = iris.data[i*subset_size:(i+1)*subset_size]
        y_test = iris.target[i*subset_size:(i+1)*subset_size]
        X_train = np.concatenate([iris.data[:i*subset_size],iris.data[(i+1)*subset_size:]])
        y_train = np.concatenate([iris.target[:i*subset_size],iris.target[(i+1)*subset_size:]])
        tree = buildTree(X_train,y_train)
        #計算錯誤率
        error += test(X_test,y_test,tree)
    #平均錯誤率
    return error/k
        
error_rate = K_fold_CV(10,iris)  
print("錯誤率:",error_rate)

