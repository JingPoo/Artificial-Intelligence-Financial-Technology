# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:15:35 2020

@author: Jing
"""

import numpy as np
import matplotlib.pyplot as plt


# t ** 4 係數為0，算出來的係數1.32158356e-07也很小，所以當然這係數沒有
def F1(t, Tc, B, fi, w):
    return 0.063 * (t ** 3) - 5.284 * (t ** 2) + 4.887 * t + 412 + np.random.normal(0, 1) # 後面是noise

def F2(A, B, C, Tc, Beta, fi, w):
    # return A * (x ** B) + C * np.cos(D * x) + np.random.normal(0, 1, x.shape) # mean = 0, std = 1
    t = 1
    return A + B * (Tc - t) ** Beta + B * C * (Tc - t) ** Beta  * np.cos((w * np.log(Tc - t) + fi))


def gene_to_coef(gene):
    A = (np.sum(2 ** np.arange(4) * gene[0 : 4]) + 500) # 2進位轉成10進位
    B = (np.sum(2 ** np.arange(-1, -9, -1, dtype=float) * gene[4 : 12]))
    C = (np.sum(2 ** np.arange(8) * gene[12 : 20]))
    D = (np.sum(2 ** np.arange(10) * gene[20:]))
    return A, B, C, D

def linear_reg(Tc, Beta, fi, w, data):
    n = Tc
    A = np.zeros((n, 3))
    b = np.zeros((n, 1))
    for i in range(Tc):
        b[i] = np.log(data[i])
        A[i, 0] = 1
        A[i, 1] = (Tc - i) ** Beta
        A[i, 2] = (Tc - i) ** Beta  * np.cos((w * np.log(Tc - i) + fi))
        
    # Ax = b
    # LeaST SQuare 線性回歸
    x = np.linalg.lstsq(A, b)[0] 
    return x[0][0], x[1][0], x[2][0]



n = 1000
# 基因演算法
T = np.random.random((n, 1)) * 100 # 先產生1000個參數
b2 = F2(1, 1, 1, 500, 0.2, 100, 1000)

pop = np.random.randint(0, 2, (10000, 30)) # 產生0~1
fit = np.zeros((10000, 1))


# 10個世代，繁衍
for generation in range(10):
    for i in range(10000):
        Tc, Beta, fi, w = gene_to_coef(pop[i, :])
        fit[i] = np.mean(abs(F2(1, 1, 1, Tc, Beta, fi, w) - b2)) # fit 越小越好，因為錯誤率約低越好
    print(np.mean(fit))
    sortf = np.argsort(fit[:, 0]) # 由小到大排，留下index
    pop = pop[sortf, :] # check一下他的列值會不會換
    # 留下前100人，後面的人被前面交配的後代取代
    for i in range(100, 10000):
        fid = np.random.randint(0, 100) # 0 ~ 99產生爸爸
        mid = np.random.randint(0, 100) # 產生媽媽
        # 怕無性生殖，爸爸等於媽媽
        while mid == fid:
            mid = np.random.randint(0, 100)
        mask = np.random.randint(0, 2, (1, 30)) # matrix:(1 * 40)
        son = pop[mid, :] # 兒子先用媽媽的資料
        father = pop[fid, :] 
        son[mask[0, :] == 1] = father[mask[0, :] == 1] # 0的基因來自媽媽，1的基因來自爸爸
        pop[i, :] = son # 把第i人用son替換掉
    # 突變1000人
    for i in range(1000):
        m = np.random.randint(0, 10000)
        n = np.random.randint(0, 30)
        pop[m ,n] = 1 - pop[m, n]


# 因為上面只得出目前最好的10000個人，但還沒排序過，就乾脆再做一次
for i in range(10000):
    A, B ,C, D = gene_to_coef(pop[i, :])
    fit[i] = np.mean(abs(F2(1, 1, 1, Tc, Beta, fi, w) - b2)) # fit 越小越好，因為錯誤率約低越好
sortf = np.argsort(fit[:, 0]) # 由小到大排，留下index
pop = pop[sortf, :] # check一下他的列值會不會換

A, B, C, D = gene_to_coef(pop[0, :]) # 取最好的那個人的資料
print(A, B, C, D) # 去跟原本的參數比較 b2 = F2(T, 0.6, 1.2, 100, 0.4)

Tc, Beta, fi, w = A, B, C, D

true_value = np.load('data.npy') # 股價的標準答案
A, B, X = linear_reg(Tc, Beta, fi, w, true_value)
C = X / B
print('A', A)
print('B', B)
print('C', C)

predict_value = np.zeros((Tc, 1))
def ln_pt(A, B, C, Tc, Beta, fi, w):
    global predict_value
    for i in range(Tc):
        predict_value[i] = A + B * (Tc - i) ** Beta + B * C * (Tc - i) ** Beta  * np.cos((w * np.log(Tc - i) + fi))
    predict_value = np.exp(predict_value[:])
    plt.plot(predict_value)
    
ln_pt(7.16, -0.43, 0.035, 9.92, 0.35, 2.07, 4.15)