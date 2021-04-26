#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#monte-carlo simulation
def MCsim(S0,T,R,vol,N):
    dt = T/N
    ST = np.random.normal(size=N+1)
    ST = np.exp((R-0.5*vol*vol)*dt+vol*ST*np.sqrt(dt))
    ST[0] = S0
    ST = np.cumprod(ST)
    return ST
sa = MCsim(50,2,0.08,0.2,200)
plt.plot(sa)
plt.show()


# In[3]:


#(1)
#模擬20000次
call=[]
for i in range(20000):
    sa = MCsim(50,2,0.08,0.2,100)
    call.append(sa[-1])
#前200次股價分布情況
plt.hist(call[:200],bins=50)
plt.xlabel('Price')
plt.ylabel('Quantity')


# In[4]:


#前2000次股價分布情況
plt.hist(call[:2000],bins=50)
plt.xlabel('Price')
plt.ylabel('Quantity')


# In[5]:


#全部模擬股價分布情況
plt.hist(call,bins=50)
plt.xlabel('Price')
plt.ylabel('Quantity')


# In[6]:


#(2)
call = []
call2 = []
for i in range(100000):
    #theta=0.2
    sa = MCsim(50,2,0.08,0.2,100)
    call.append(sa[-1])
    #theta=0.4
    sa2 = MCsim(50,2,0.08,0.4,100)
    call2.append(sa2[-1])

plt.plot(call,alpha=0.5,color='r')
plt.plot(call2,alpha=0.5,color='g')
plt.xlabel('Simulation times')
plt.ylabel('ST')


# In[7]:


#black-scholes model
def callblsprice(S,K,r,T,vol): #vol:波動率
    d1 = (math.log(S/K)+(r+vol*vol/2)*T)/(vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    call = S*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2)
    return call


# In[8]:


#(3)
bls = callblsprice(50,40,0.08,2,0.2)
N = [100,1000,10000]

for i in N:
    print('分',i,'期')
    call=0
    for j in range(10000):
        sa = MCsim(50,2,0.08,0.2,i)
        if sa[-1]>40:
            call += (sa[-1]-40)
        if j==99:
            x = call
            print('前100次與bls model的絕對誤差:',abs((x/100)*math.exp(-0.08*2)-bls))
        elif j==999:
            y = call
            print('前1000次與bls model的絕對誤差:',abs((y/1000)*math.exp(-0.08*2)-bls))
        elif j==9999:
            z = call
            print('全部與bls model的絕對誤差:',abs((z/10000)*math.exp(-0.08*2)-bls))


# In[9]:


#Binomial Model
def BTcall(S0,K,T,R,vol,N):  #N:depth of tree
    dt = T/N
    u = math.exp(vol*math.sqrt(dt))
    d = 1/u
    p = (math.exp(R*dt)-d) / (u-d)
    priceT = np.zeros((N+1,N+1))
    priceT[0][0] = S0
    for c in range(N):
        priceT[0][c+1] = priceT[0][c]*u #每一column，row1往右一直*u
        for r in range(c+1):
            priceT[r+1][c+1] = priceT[r][c]*d #每一column，各row往右下*d
    probT = np.zeros((N+1,N+1))
    probT[0][0]=1
    for c in range(N):
        for r in range(c+1):
            probT[r][c+1] += probT[r][c]*p #往右*p
            probT[r+1][c+1] += probT[r][c]*(1-p) #往右下*(1-p)
    call = 0
    for r in range(N+1): #模擬出的100個價位中
        if priceT[r][N]>K: #如果價值比K高(沒比K高沒價值)
            call += (priceT[r][N]-K)*probT[r][N] #把價值乘上機率加進CALL
    return call*math.exp(-R*T) #回推現在應該要的價格

#print(BTcall(50,40,2,0.08,0.2,100)) 


# In[10]:


(4)
print('Binomial Model:')
print('N=10:',BTcall(50,40,2,0.08,0.2,10))
print('N=100:',BTcall(50,40,2,0.08,0.2,100))
print('N=1000:',BTcall(50,40,2,0.08,0.2,1000))
print('N=10000:',BTcall(50,40,2,0.08,0.2,10000))
#print('N=50000:',BTcall(50,40,2,0.08,0.2,50000))
print('black-scholes model:')
print(callblsprice(50,40,0.08,2,0.2))
    


# In[11]:


(5)
#亞式選擇權
call = 0
for i in range(1000):
    sa = MCsim(50,2,0.08,0.2,100)
    if sa.mean()>40: #若中間一段時間的平均比40大
        call += (sa[-1]-40)/1000
print(call*math.exp(-0.08*2))

