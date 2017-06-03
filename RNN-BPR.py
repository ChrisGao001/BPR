# -*- coding:utf-8 -*-
# item:8533
# user:733

import numpy as np
import random
import json
import sys
import copy
import multiprocessing

class hps(object):
    def __init__(self,
                 filename='data/user_cart.json',
                 num_test=1,
                 learning_rate=0.02,
                 lamda_pos=0.001,
                 lamda=0.001,
                 hidden_size=10,
                 num_unrolling=3):
        if filename == None:
            raise ValueError()
        self.filename = filename
        self.num_test = num_test
        self.learning_rate = learning_rate
        self.lamda_pos = lamda_pos
        self.lamda = lamda
        self.hidden_size = hidden_size
        self.num_unrolling = num_unrolling


class rnn(object):
    def __init__(self, hps):
        self.hps = hps
        self._load_data()
        self.reset_train_state()
        self.reset_state()

    def _load_data(self):
        self.data = []
        self.neg = []
        self.num_user = 0
        with open(self.hps.filename, 'r') as f:
            for line in f:
                linedata = json.loads(line)
                if len(linedata) < 10:
                    continue
                self.num_user += 1
                self.data.append([int(i) for i in linedata])
                tmp = []
                for i in linedata:
                    while True:
                        j = random.randint(0, 8532)
                        if j != int(i - 1):
                            break
                    tmp.append(j)
                self.neg.append(tmp)

    def reset_train_state(self):
        self.Wx = np.random.randn(
            self.hps.hidden_size, self.hps.hidden_size) * 0.5
        self.Wh = np.random.randn(
            self.hps.hidden_size, self.hps.hidden_size) * 0.5
        self.item = np.random.randn(8533, self.hps.hidden_size) * 0.5

    def reset_state(self):
        self.h = np.zeros((self.hps.num_unrolling + 1, self.hps.hidden_size))
        self.idx_inps = -np.ones(self.hps.num_unrolling, dtype=int)
        self.sigmoid_d = np.zeros(
            (self.hps.num_unrolling + 1, self.hps.hidden_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self):
        for n in range(self.num_user):
            self.reset_state()
            # for step in range(int(len(self.data[n])*0.8-1)):
            for step in range(len(self.data[n])-self.hps.num_unrolling-1):
                sys.stdout.write('Training user: %5d, item: %5d\r' % (n, step))
                sys.stdout.flush()
                self.train_step(self.data[n][step] - 1,
                                self.data[n][step + 1] - 1,
                                self.neg[n][step + 1],
                                step)

    def train_step(self, idx_inp, idx_pos, idx_neg, step):
        self.idx_inps[1:] = self.idx_inps[:-1]
        self.idx_inps[0] = idx_inp

        self.h[1:] = self.h[:-1]
        self.sigmoid_d[1:] = self.sigmoid_d[:-1]
        self.h[0] = self.sigmoid(
            np.dot(self.item[self.idx_inps[0]], self.Wx) + np.dot(self.h[1], self.Wh))
        self.sigmoid_d[0] = self.h[0] * (1 - self.h[0])

        Xij = np.dot(self.h[0], self.item[idx_pos]) - \
            np.dot(self.h[0], self.item[idx_neg])

        share = self.sigmoid(-Xij)

        self.item[idx_pos] += self.hps.learning_rate * \
            (share * self.h[0] - self.hps.lamda * self.item[idx_pos])
        self.item[idx_neg] += self.hps.learning_rate * \
            (share * (-self.h[0]) - self.hps.lamda * self.item[idx_neg])

        for ustep in range(min(step + 1, self.hps.num_unrolling)):
            if ustep == 0:
                product = self.item[idx_pos] - self.item[idx_neg]
            else:
                product = np.dot(self.sigmoid_d[0] * product, self.Wh.T)
            if step == 0:
                dWx = 0
                dWh = 0
            else:
                dWx = np.dot(self.item[self.idx_inps[ustep]].reshape(
                    (1, -1)).T, (self.sigmoid_d[ustep] * product).reshape((1, -1)))
                dWh = np.dot(self.h[ustep + 1].reshape((1, -1)).T,
                             (self.sigmoid_d[ustep] * product).reshape((1, -1)))
            self.Wx += self.hps.learning_rate * \
                (share * dWx - self.hps.lamda * self.Wx)
            self.Wh += self.hps.learning_rate * \
                (share * dWh - self.hps.lamda * self.Wh)

    def eval(self):
        num_pred = 0
        all_pred = 0
        for n in range(self.num_user):
            self.reset_state()
            for t in range(len(self.data[n]) - 1):
                # sys.stdout.write('Predicting user: %5d, item: %5d\r' % (n, t))
                # sys.stdout.flush()
                if t < len(self.data[n]) - self.hps.num_test:
                # if t < int(len(self.data[n])*0.8):
                    pred = self.prediction(self.item[self.data[n][t] - 1])
                else:
                    pred = self.prediction(pred)
                if t > len(self.data[n]) - self.hps.num_test-2:
                # if t > int(len(self.data[n])*0.8)-2:
                    tmp = np.dot(pred, self.item.T)
                    sort_list = np.argsort(-tmp)
                    for k in range(10):
                        if sort_list[k] == self.data[n][t + 1] - 1:
                            num_pred += 1
                            break
                    all_pred += 1
        return num_pred, all_pred


    def prediction(self, input):
        self.h[0] = self.sigmoid(
            np.dot(input, self.Wx) + np.dot(self.h[0], self.Wh))
        return self.h[0]

def eval_async(R):
    num_p, all_p = R.eval()
    print(str(num_p / all_p)+'                                   ')

R = rnn(hps())
step=0
# for step in range(500):
while True:
    R.train()
    print('Step %d done.                               ' % step)
    p=multiprocessing.Pool()
    R_eval=copy.deepcopy(R)
    p.apply_async(eval_async,(R_eval,))
    p.close()
    step+=1
print('Dnoe.')
