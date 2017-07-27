#!/usr/bin/env python
# encoding:utf-8

#Author: Mi Zhangpeng
import math
import numpy as np
import pandas as pd
import random

class neural: #组成网络的神经元
    def __init__(self):
        self.inlinks = []
        self.weights = []
        self.threshold = 0  
        self.output = None

class neural_network:  #单隐层网络
    def __init__(self):
        self.input_layer = []
        self.hidden_layer = []
        self.output_layer = []


def rand():   #生成(0,1)的随机数
    value = random.random()
    while value == 0:
        value = random.random()
    return value

def buildNetwork():  #生成由固定神经元个数组成的网络
    network = neural_network()
    for i in range(8):
        input_neural = neural()
        network.input_layer.append(input_neural)

    for i in range(10):
        hidden_neural = neural()
        for it in network.input_layer:
            hidden_neural.inlinks.append(it)
            hidden_neural.weights.append(rand())
        hidden_neural.threshold = rand()
        network.hidden_layer.append(hidden_neural)
 
    
    output_neural = neural()
    for it in network.hidden_layer:
        output_neural.inlinks.append(it)
        output_neural.weights.append(rand())
    output_neural.threshold = rand()
    network.output_layer.append(output_neural)
    return network


def neural_func(neural):  #神经元处理输入
    inputsum = 0
    for i in range(len(neural.inlinks)):
        inputx = neural.inlinks[i].output
        weight = neural.weights[i]
        inputsum += inputx*weight
    neural.output = 1/(1+math.exp(-(inputsum-neural.threshold)))

def cal_network_output(x,network):  #根据输入计算神经网络的输出
    for i in range(len(x)):
        network.input_layer[i].output = x[i]
    for i in range(len(network.hidden_layer)):
        neural_func(network.hidden_layer[i])
    for i in range(len(network.output_layer)):
        neural_func(network.output_layer[i])
    return [network.output_layer[0].output]
        
def back_propagation(network, data, eta, error):  #标准BP算法
    roundnum = 0
    while True:
        roundnum += 1
        E = 0
        for k in range(data.shape[0]):
            example = data.iloc[k]
            x = example[:-1]
            y = example[-1:]
            yhats = cal_network_output(x,network)
            Ek = 0
            for j in range(len(yhats)):
                Ek += (y[j]-yhats[j])**2
            Ek = Ek/2
            E += Ek
            g = [0]*len(network.output_layer)
            e = [0]*len(network.hidden_layer)
            for j in range(len(network.output_layer)):
                output_neural = network.output_layer[j]
                g[j] = yhats[j]*(1-yhats[j])*(y[j]-yhats[j])
                for h in range(len(output_neural.weights)):
                    bh = output_neural.inlinks[h].output
                    e[h] += bh*(1-bh)*output_neural.weights[h]*g[j]
                    output_neural.weights[h] += eta*g[j]*bh
                output_neural.threshold += -eta*g[j]
            for h in range(len(network.hidden_layer)):
                hidden_neural = network.hidden_layer[h]
                for i in range(len(hidden_neural.weights)):
                    hidden_neural.weights[i] += eta*e[h]*hidden_neural.inlinks[i].output
                hidden_neural.threshold += -eta*e[h]
        E = E/data.shape[0]
        if E < error:
            break
    print "训练轮数：",roundnum

def accumulated_back_propagation(network, data, eta, error): #累积BP算法
    roundnum = 0
    while True:
        roundnum += 1 #训练轮数
        m = data.shape[0] #训练样例个数
        E = 0 #累积误差
        deltaw = [[0]*len(network.output_layer) for i in range(len(network.hidden_layer))] #存放隐层和输出层链接由所有样例累积的权值变化
        deltav = [[0]*len(network.hidden_layer) for i in range(len(network.input_layer))] #存放输入层和隐层链接由所有样例累积的权值变化
        deltatheta = [0]*len(network.output_layer) #存放输出层一轮累积的阈值变化
        deltagamma = [0]*len(network.hidden_layer) #存放隐层一轮累积的阈值变化
        #计算累积的权值和阈值变化
        for k in range(m):
            example = data.iloc[k]
            x = example[:-1]
            y = example[-1:]
            yhats = cal_network_output(x,network)
            Ek = 0
            for j in range(len(yhats)):
                Ek += (y[j]-yhats[j])**2
            Ek = Ek/2
            E += Ek
            g = [0]*len(network.output_layer)
            e = [0]*len(network.hidden_layer)
            for j in range(len(network.output_layer)):
                output_neural = network.output_layer[j]
                g[j] = yhats[j]*(1-yhats[j])*(y[j]-yhats[j])
                for h in range(len(output_neural.weights)):
                    bh = output_neural.inlinks[h].output
                    e[h] += bh*(1-bh)*output_neural.weights[h]*g[j]
                    deltaw[h][j] += eta*g[j]*bh
                deltatheta[j] += -eta*g[j]
            for h in range(len(network.hidden_layer)):
                hidden_neural = network.hidden_layer[h]
                for i in range(len(hidden_neural.weights)):
                    deltav[i][h] += eta*e[h]*hidden_neural.inlinks[i].output
                deltagamma[h] += -eta*e[h]
        #对所有权值和阈值进行更新
        for j in range(len(network.output_layer)):
            output_neural = network.output_layer[j]
            for h in range(len(output_neural.weights)):
                output_neural.weights[h] += deltaw[h][j]/m
            output_neural.threshold += deltatheta[j]/m
        for h in range(len(network.hidden_layer)):
            hidden_neural = network.hidden_layer[h]
            for i in range(len(hidden_neural.weights)):
                hidden_neural.weights[i] += deltav[i][h]/m
            hidden_neural.threshold += deltagamma[h]/m
        E = E/m
        if E < error:
            break
    print "训练轮数：",roundnum            

if __name__ == "__main__":
    index = ["色泽","根蒂","敲声","纹理","脐部","触感","密度","含糖率","类别"]
    valuetypes = pd.Series([1,1,1,1,1,1,0,0,1],index=index)
    aval_values = pd.Series([["浅白","青绿", "乌黑"],["硬挺","稍蜷", "蜷缩"],["沉闷","浊响", "清脆"],
                             ["模糊","稍糊", "清晰"],["凹陷","平坦","稍凹"],["硬滑","软粘"],[],[],["好瓜","坏瓜"]],index=index)
    data = pd.read_csv("melon_data.txt",header=None,names=index)
##    print data[(data["类别"]=="好瓜")&(data["密度"]>0.5)].loc[:,["色泽","根蒂"]]
    data["色泽"].replace(["浅白","青绿", "乌黑"],[0,0.5,1],inplace=True)
    data["根蒂"].replace(["硬挺","稍蜷", "蜷缩"],[0,0.5,1],inplace=True)
    data["敲声"].replace(["沉闷","浊响", "清脆"],[0,0.5,1],inplace=True)
    data["纹理"].replace(["模糊","稍糊", "清晰"],[0,0.5,1],inplace=True)
    data["脐部"].replace(["平坦","稍凹", "凹陷"],[0,0.5,1],inplace=True)
    data["触感"].replace(["硬滑","软粘"],[0,1.0],inplace=True)
    data["类别"].replace(["坏瓜","好瓜"],[0,1.0],inplace=True)
    train_data = pd.concat([data.iloc[:5],data.iloc[8:13]])
    network = buildNetwork()
    input_data = data.iloc[8]
    
    print "标准BP算法训练的网络,误差为0.012"
    back_propagation(network,train_data,0.1,0.012)
    print "测试一条数据:"
    print input_data
    print "测试结果为:"
    print cal_network_output(input_data[:-1],network)
    print " "
    
    print "累积BP算法训练的网络,误差为0.012"
    accumulated_back_propagation(network,train_data,0.1,0.012)
    print "测试一条数据:"
    print input_data
    print "测试结果为:"
    print cal_network_output(input_data[:-1],network)

    print "累积BP算法训练的网络,误差为0.013"
    accumulated_back_propagation(network,train_data,0.1,0.013)
    print "测试一条数据:"
    print input_data
    print "测试结果为:"
    print cal_network_output(input_data[:-1],network)
