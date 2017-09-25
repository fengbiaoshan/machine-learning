#!/usr/bin/env python
# coding:utf-8

#Author: Mi Zhangpeng

#给出方程F(x) = 6*(x**7) + 8*(x**6) + 7*(x**3) + 5*(x**2) -xy，其中0<=x<=100，输入y，求F(x)的最小值

import random
import math

iters = 10
T = 100
eps = 1e-8
delta = 0.98
inf = 1e99

def rand():
    return random.uniform(-1,1)

def F(x, y):
    return 6*(x**7) + 8*(x**6) + 7*(x**3) + 5*(x**2) - x*y

def SA(y):
    x = abs(rand())*100
    ans = inf
    t = T
    while t > eps:
        tmp = F(x, y)
        for j in range(iters):
            xx = x+rand()*t
            if xx >= 0 and xx <= 100:
                f = F(xx,y)
                de = tmp - f
                if de > 0:
                    x = xx
                else:
                    if (math.exp(de/t) > random.random()):
                        x = xx
        t *= delta
    return F(x,y)

if __name__ == "__main__":
    y = 100
    print "%.4lf" % SA(y)
