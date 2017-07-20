#!/usr/bin/env python
# encoding:utf-8

#Author: Mi Zhangpeng
import math
import numpy as np
import pandas as pd
from collections import deque


class decision_tree_node:
    def __init__(self,attr,children):
        self.attr = attr   #结点依据的划分属性，叶节点为类别
        self.children = children
    def __repr__(self):
        return self.attr
    

def information_entropy(data,total):
    result = 0
    total = np.float64(total)
    classcounts = data.iloc[:,-1].value_counts()
    for it in classcounts:
        result -= (it/total)*math.log((it/total),2)
    return result


def information_gain(data,attr):
    datacount = np.float64(data.iloc[:,-1].count())
    if valuetypes[attr] == 0:
        valuecounts = data[attr].value_counts()
        valuecounts.sort_index(inplace=True)
        min_entropy_sum = 1
        opt_t = 0
        for i in xrange(len(valuecounts)-1):
            t = (valuecounts.index[i] + valuecounts.index[i+1])/2
            data1 = data[data[attr]<=t]
            count1 = data1.iloc[:,-1].count()
            data2 = data[data[attr]>t]
            count2 = data2.iloc[:,-1].count()
            entropy_sum = 0
            entropy_sum += (count1/datacount)*information_entropy(data1,count1)
            entropy_sum += (count2/datacount)*information_entropy(data2,count2)
            if entropy_sum < min_entropy_sum:
                min_entropy_sum = entropy_sum
                opt_t = t
        return (information_entropy(data,datacount) - min_entropy_sum,opt_t)        
    else:
        entropy_sum = 0
        valuecounts = data[attr].value_counts()
        for value,count in valuecounts.iteritems():
            datav = data[data[attr]==value]
            entropy_sum += (count/datacount)*information_entropy(datav,count)
        return (information_entropy(data,datacount) - entropy_sum,None)
    
    
def decision_tree_generate(data,attrset):
    classcount = data.iloc[:,-1].value_counts()
    if len(classcount) == 1:
        node = decision_tree_node(classcount.keys()[0],None)
        return node
    sameinattrset = True
    for it in attrset:
        if len(data[it].value_counts()) != 1:
            sameinattrset = False
            break
    if not attrset or sameinattrset:
        node = decision_tree_node(classcount.idxmax(),None)
        return node
    optattr = None
    opt_attr_gain = 0
    opt_t = None
    for attr in attrset:
        attr_gain,t = information_gain(data,attr)
        if attr_gain > opt_attr_gain:
            optattr = attr
            opt_t = t
            opt_attr_gain = attr_gain
    children = {}
    node = decision_tree_node(optattr,children)
    if valuetypes[optattr] == 0:
        datav1 = data[data[optattr]<=opt_t]
        if datav1.empty:
            childnode = decision_tree_node(classcount.idxmax(),None)
        else:
            childnode = decision_tree_generate(datav1,attrset)
        children[-opt_t] = childnode
        datav2 = data[data[optattr]>opt_t]
        if datav2.empty:
            childnode = decision_tree_node(classcount.idxmax(),None)
        else:
            childnode = decision_tree_generate(datav2,attrset)
        children[opt_t] = childnode
        return node
    else:
        for value in aval_values[optattr]:
            datav = data[data[optattr]==value]
            if datav.empty:
                childnode = decision_tree_node(classcount.idxmax(),None)
                children[value] = childnode
            else:
                attrset.remove(optattr)
                childnode = decision_tree_generate(datav,attrset)
                attrset.add(optattr)
                children[value] = childnode
        return node

def decision_tree_test(input_data,tree_root):
    node = tree_root
    while node.children != None:
        attrvalue = input_data[node.attr]
        if valuetypes[node.attr] == 0:
            if attrvalue < abs(node.children.keys()[0]):
                node = node.children[-node.children.keys()[0]]
            else:
                node = node.children[node.children.keys()[0]]
        else:
            node = node.children[attrvalue]
    return node.attr

def print_decision_tree(root):
    queue = deque()
    queue.append(root)
    queue.append(None)
    tmp = ""
    while len(queue) != 1:
        item = queue.popleft()
        if item == None:
            print tmp.decode("utf-8")
            tmp = ""
            queue.append(None)
        elif item == "":
            tmp += "()"
        else:
            tmp += "("+item.attr
            if item.children != None:
                for key,value in item.children.items():
                    queue.append(value)
                    tmp += ","+str(key)
            else:
                queue.append("")
            tmp += ")"
    print tmp.decode("utf-8")
    

if __name__ == "__main__":
    index = ["色泽","根蒂","敲声","纹理","脐部","触感","密度","含糖率","类别"]
    valuetypes = pd.Series([1,1,1,1,1,1,0,0,1],index=index)
    aval_values = pd.Series([["青绿","乌黑","浅白"],["蜷缩","稍蜷","硬挺"],["浊响","沉闷","清脆"],
                             ["清晰","稍糊","模糊"],["凹陷","平坦","稍凹"],["硬滑","软粘"],[],[],["好瓜","坏瓜"]],index=index)
    data = pd.read_csv("melon_data.txt",header=None,names=index)
##    print data[(data["类别"]=="好瓜")&(data["密度"]>0.5)].loc[:,["色泽","根蒂"]]
    attrset = set(index[:-1])
    train_data = pd.concat([data.iloc[:5],data.iloc[8:13]])
    root = decision_tree_generate(train_data,attrset)
##    print_decision_tree(root)
    input_data = data.iloc[5]
    print "测试一条数据:"
    print input_data
    print "测试结果为:"
    print decision_tree_test(input_data,root).decode("utf-8")
