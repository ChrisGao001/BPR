#user_size 707
#item_size 8533

import numpy as np
import random
import math
import json
import sys

data = []
training_set = []
raw_data = open('data/user_cart.json', 'r')
lines = raw_data.readlines()
count = 0
for line in lines:
    line1 = json.loads(line)
    n = 0
    for num in line1:
        line1[n] = int(line1[n])
        n += 1
    if n >= 10:
        data.append(line1)
        training_set.append(line1[0:int(n*0.8)])
        count += 1

u = np.random.randn(10, 10)*0.5
r = np.random.randn(10, 10)*0.5
i = np.random.randn(8533, 10)*0.5 

learning_rate = 0.01
lamda_pos = 0.001
lamda = 0.001
hidden_size = 10

def f(x):    #sigmoid
	output = 1/(1+np.exp(-x))
	return output
	
neglist = []
for n in range(707):
    line = []
    for num in training_set[n]:
        j = random.randint(0, 8532)
        while j == num:
            j = random.randint(0, 8532)
        line.append(j)
    neglist.append(line)
    
def train():
    global u
    global r
    global i
    for n in range(707):
        sys.stdout.write('Training %5d\r' % n)
        sys.stdout.flush()
        count = 0
        for num in training_set[n]:
            count += 1
        h=np.zeros((10,10))
        idx_inp=np.zeros(10,dtype=int)
        sigmoid_d=np.zeros((10,10))
        for step in range(count-1):
            idx_inp[1:] = idx_inp[:-1]

            idx_inp[0] = training_set[n][step]-1
            idx_pos = training_set[n][step+1]-1
            idx_neg = neglist[n][step+1]-1

            h[1:]=h[:-1]
            sigmoid_d[1:]=sigmoid_d[:-1]
            h[0]=f(np.dot(i[idx_inp[0]], u) + np.dot(h[1], r))
            sigmoid_d[0]=h[0]*(1-h[0])

            x = np.dot(h[0], i[idx_pos]) - np.dot(h[0], i[idx_neg])

            mid = 1/ (1 + np.exp(x))

            i[training_set[n][step+1]-1] += learning_rate * (mid * h[0] - lamda * i[idx_pos])
            i[neglist[n][step+1]] += learning_rate * (mid * (-h[0]) - lamda * i[idx_neg])

            for ustep in range(min(step+1,3)):
                if ustep==0:
                    product = i[idx_pos] - i[idx_neg]
                else:
                    product = np.dot(sigmoid_d[0] * product, r.T)
                if step==0:
                    xu = 0
                    xr = 0
                else:
                    xu=np.dot(i[idx_inp[ustep]].reshape((1,-1)).T,(sigmoid_d[ustep] * product).reshape((1,-1)))
                    xr=np.dot(h[ustep+1].reshape((1,-1)).T,(sigmoid_d[ustep] * product).reshape((1,-1)))
                u += learning_rate * (mid * xu - lamda * u)
                r += learning_rate * (mid * xr - lamda * r)
def predict_next(h,inp):
    return f(np.dot(h,r) + np.dot(inp,u))
def predict():
    predict_count = 0
    predict_sum = 0
    n = 0
    while n < 707:
        sys.stdout.write('Predicting %5d\r' % n)
        sys.stdout.flush()
        count = 0
        for num in data[n]:
            count += 1
        h = np.zeros((1, 10))
        if count > 1:
            for m in range(int(count*0.8)):
                inp = i[data[n][m]-1]
                h=predict_next(h,inp)
            while m < (count - 1):
                temp = np.dot(h, i.T)
                sort_list = np.argsort(-temp[0])
                for k in range(10):
                    if (sort_list[k]+1) == data[n][m+1]:
                        predict_count += 1
                predict_sum += 1
                inp = i[data[n][m+1]-1]
                h=predict_next(h,inp)
                m += 1
        n += 1
    return predict_count, predict_sum
    
for n in range(200):
    print(n)
    train()
    predict_count, predict_sum = predict()
    result = float(predict_count) / predict_sum   
    print(result)
 
