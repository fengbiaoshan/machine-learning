#!/usr/bin/env python
# coding:utf-8

#Author: Mi Zhangpeng
import random
import math
import numpy as np
import pandas as pd

def lr_newton(xlist, ylist, origbeta, n):
    xhatlist = [] #xhat = (x;1)
    for x in xlist:
        xhat = np.append(x,[[1]],axis=0)
        xhatlist.append(xhat)
    beta = origbeta
    for i in xrange(n):
        beta = beta - np.dot(np.linalg.inv(hesse(xhatlist,beta)),jacobi(xhatlist,ylist,beta))
    return beta

def jacobi(xhatlist,ylist,beta):
    n = xhatlist[0].shape[0]
    result = np.array([[0]*n]).T
    for i in xrange(len(xhatlist)):
        xhat = xhatlist[i]
        p1 = math.exp(np.dot(beta.T,xhat)[0][0])/(math.exp(np.dot(beta.T,xhat)[0][0])+1)
        result = result + np.dot(xhat,(ylist[i]-p1))
    return -result

def hesse(xhatlist,beta):
    n = xhatlist[0].shape[0]
    result = np.array([[0]*n for i in range(n)])
    for i in xrange(len(xhatlist)):
        xhat = xhatlist[i]
        p1 = math.exp(np.dot(beta.T,xhat)[0][0])/(math.exp(np.dot(beta.T,xhat)[0][0])+1)
        result = result + np.dot(np.dot(xhat,xhat.T),p1*(1-p1))
    return result


def linear_classifier(beta, x):
    xhat = np.append(x,[[1]],axis=0)
    y = math.exp(np.dot(beta.T,xhat)[0][0])/(math.exp(np.dot(beta.T,xhat)[0][0])+1)
    if y > 0.5:
        return 1
    elif y == 0.5:
        return random.randint(0,1)
    else:
        return 0


def k_fold_cross_validation(data, k):
    errorproall = 0.0
    data_number = data.shape[0]
    xlist = np.hsplit(data.iloc[:,:-1].values.T, data_number)
    ylist = []
    for it in data.iloc[:,-1].values:
        if it == data.iloc[0,-1]:
            ylist.append(0)
        else:
            ylist.append(1)
    for i in range(1,k+1):
        xtrainlist =xlist[:data_number/2/k*(i-1)] + xlist[data_number/2/k*(i):data_number/2+data_number/2/k*(i-1)] + xlist[data_number/2+data_number/2/k*i:]
        ytrainlist = ylist[:data_number/2/k*(i-1)] + ylist[data_number/2/k*(i):data_number/2+data_number/2/k*(i-1)] + ylist[data_number/2+data_number/2/k*i:]
        origbeta = np.array([[0]*(xlist[0].shape[0]+1)]).T
        origbeta[-1] = 1
        print len(xtrainlist),len(ytrainlist)
        beta = lr_newton(xtrainlist,ytrainlist,origbeta,50)
        xtestlist = xlist[data_number/2/k*(i-1):data_number/2/k*(i)] + xlist[data_number/2+data_number/2/k*(i-1):data_number/2+data_number/2/k*i]
        ytestlist = ylist[data_number/2/k*(i-1):data_number/2/k*(i)] + ylist[data_number/2+data_number/2/k*(i-1):data_number/2+data_number/2/k*i]
        print len(xtestlist),len(ytestlist)
        errorcount = 0.0
        for i in xrange(len(xtestlist)):
            if linear_classifier(beta, xtestlist[i]) != ytestlist[i]:
                errorcount += 1
        errorproall += errorcount/k
    
    return errorproall/k

if __name__ == "__main__":
    data = pd.read_csv("iris.data",header=None,names=["sl","sw","pl","pw","class"])
    data = data.iloc[:100,:]
    print k_fold_cross_validation(data,10)
    


    
