# encoding:gbk
#!/usr/bin/env python
import math
from numpy import *

#计算类内散度矩阵
def wcsm(xlist,ylist,u0,u1):
    n = xlist[0].shape[0]
    result = array([[0]*n for i in range(n)])
    for i in xrange(len(xlist)):
        x = xlist[i]
        if ylist[i] == 0:
            result = result + dot((x-u0),(x-u0).T)
        else:
            result = result + dot((x-u1),(x-u1).T)
    return result
        

if __name__ == "__main__":
    xlist = []
    ylist = []
    f = open("./melon_data.txt", "r")
    line = f.readline()
    while line:
        linelist = line.split()
        x = array([[]])
        for i in range(1,len(linelist)-1):
            x = append(x,[[float(linelist[i])]],axis=1)
        xlist.append(x.T)
        ylist.append(float(linelist[-1]))
        line = f.readline()
    f.close()
    n = n = xlist[0].shape[0]
    xsum0 = array([[0]*n]).T
    xsum1 = array([[0]*n]).T
    count0 = 0
    count1 = 0
    for i in xrange(len(xlist)):
        x = xlist[i]
        if ylist[i] == 0:
            xsum0 = xsum0 + x
            count0 += 1
        else:
            xsum1 = xsum1 + x
            count1 += 1
    u0 = xsum0/count0
    u1 = xsum1/count1
    Sw = wcsm(xlist,ylist,u0,u1)
    #利用奇异值分解算Sw的逆矩阵
    U,sigma,vt=linalg.svd(Sw)
    sigmainv = 1/sigma
    sigmainv = diag(sigmainv)
    Swinv = dot(dot(vt.T, sigmainv),U.T)
    w = dot(Swinv,u0-u1)
    print w  #求出的线性系数
    






    
        

