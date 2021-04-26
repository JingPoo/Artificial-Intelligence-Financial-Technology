# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#房貸計算器
print("歡迎使用房貸計算器! 請輸入\n")
print("貸款總額")
total = int(input())
print("\n貸款期限")
T = int(input())
print("\n寬限期")
N = int(input())
r = 0.0133 #年利率
T*=12
N*=12


#本息平均攤還
print("\n玉山商業銀行(1.33%):\n本息平均攤還\n")
for j in range(1,T+1):
    if j <= N:
        monthly_pay = round(total*(r/12)) #寬限期間每月還的錢
    else:
        n = T-N
        monthly_pay = round( total*(1+r/12)**n*(r/12) / ((1+r/12)**n-1) ) #寬限期後每月總共要還的錢
    print(j,": $",monthly_pay)
    
#本金平均攤還
print("\n本金平均攤還\n")
for i in range(1,T+1):
    if i <= N:
        monthly_pay = round(total*(r/12)) #寬限期間每月還的錢
    else:
        x = total/(T-N) #寬限期後每月應繳本金
        y = (total-x*(i-N-1))*(r/12) #寬限期後每月應繳利息
        monthly_pay = round(x+y)
    print(i,": $",monthly_pay)
    


        