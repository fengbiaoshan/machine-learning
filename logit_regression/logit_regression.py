# encoding:gbk
#!/usr/bin/env python
import math
from numpy import *

#origbeta为使用牛顿迭代法的beta初始值(初始值需要人为指定)，beta = (x;1)，是自变量x向量扩充一维。n为迭代的次数，次数越高越精确
def lr_newton(xlist, ylist, origbeta, n):
    xhatlist = []
    for x in xlist:
        xhat = append(x,[[1]],axis=0)
        xhatlist.append(xhat)
    beta = origbeta
    for i in xrange(n):
        beta = beta - dot(linalg.inv(hesse(xhatlist,beta)),jacobi(xhatlist,ylist,beta))
    return beta

def jacobi(xhatlist,ylist,beta):
    n = xhatlist[0].shape[0]
    result = array([[0]*n]).T
    for i in xrange(len(xhatlist)):
        xhat = xhatlist[i]
        p1 = math.exp(dot(beta.T,xhat)[0][0])/(math.exp(dot(beta.T,xhat)[0][0])+1)
        result = result + dot(xhat,(ylist[i]-p1))
    return -result

def hesse(xhatlist,beta):
    n = xhatlist[0].shape[0]
    result = array([[0]*n for i in range(n)])
    for i in xrange(len(xhatlist)):
        xhat = xhatlist[i]
        p1 = math.exp(dot(beta.T,xhat)[0][0])/(math.exp(dot(beta.T,xhat)[0][0])+1)
        result = result + dot(dot(xhat,xhat.T),p1*(1-p1))
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
    origbeta = array([[1]*(xlist[0].shape[0]+1)]).T
    print lr_newton(xlist,ylist,origbeta,100)
        

