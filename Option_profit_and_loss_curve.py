#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

#買權
def callr(K,call): #K:履約價 call:花了多少錢買選擇權
    ST = np.arange(12500,13301)
    return np.maximum(ST-K,0) - call 

#賣權
def putr(K,put):
    ST = np.arange(12500,13301)
    return np.maximum(K-ST,0) - put


# In[2]:


# (1)
'''用履約價為12800，12900，13000三種買權排列組合出不同的bull spread，
並分別用紅、綠、藍三種不同顏色的線畫出損益曲線，並比較三者在不同到期價時的優缺點
'''
x1 = np.arange(12500,13301)
y1 = callr(12800,199)
y2 = callr(12900,130)
y3 = callr(13000,92)

plt.plot(x1,y1-y2,'r',x1,y1-y3,'g',x1,y2-y3,'b',[x1[0],x1[-1]],[0,0],'--k')


# In[3]:


#(2)
'''假設加權股價持續盤整，直到到期日前均會在12900附近震盪，
A.	使用履約價為12900的買賣權建構straddle
B.	使用履約價為12800和13000的買賣權建構strangle
分別用紅線及綠線繪製兩者的損益曲線，並簡述其優缺點
'''
x1 = np.arange(12500,13301)
y1 = callr(12900,130)
y2 = putr(12900,174)
y3 = callr(13000,92)
y4 = putr(12800,131)

plt.plot(x1,-y1-y2,'r',x1,-y3-y4,'g',[x1[0],x1[-1]],[0,0],'--k')


# In[4]:


#(3)
'''
用履約價相同的買賣權，可以組合出等同於放空一口指數的商品
A.	請使用履約價分別為12600及13200的買賣權建構此組合，以紅線及綠線繪製損益曲線，試比較兩者之優缺點
B.	在這四口選擇權中，存在套利空間，分別說明需買或賣哪幾口選擇權
'''
x1 = np.arange(12500,13301)
y1 = callr(12600,130)
y2 = putr(12600,73)
y3 = callr(13200,33)
y4 = putr(13200,373)

plt.plot(x1,y2-y1,'r',x1,y4-y3,'g',[x1[0],x1[-1]],[0,0],'--k')


# In[5]:


#(4)
'''
請用三種不同履約價的買權，組合出預期市場盤整時的butterfly spread，並簡述其應用情境
'''
x1 = np.arange(12500,13301)
y1 = callr(12800,199)
y2 = callr(12900,130)
y3 = callr(13000,92)

plt.plot(x1,y1+y3-y2*2,'r',[x1[0],x1[-1]],[0,0],'--k')


# In[ ]:




