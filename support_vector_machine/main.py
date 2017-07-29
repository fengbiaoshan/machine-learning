#!/usr/bin/env python
# encoding:utf-8

#Author: Mi Zhangpeng
import svmutil
        

if __name__ == "__main__":
    y,x = svmutil.svm_read_problem("melon_data.txt")
    #线性核
    model = svmutil.svm_train(y,x,'-t 0 -c 8.0')
    test_lable,test_acc,test_val = svmutil.svm_predict(y,x,model)

    #高斯核
    model = svmutil.svm_train(y,x,'-t 2 -c 8.0 -g 2')
    test_lable,test_acc,test_val = svmutil.svm_predict(y,x,model)

