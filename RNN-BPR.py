# item:8533
# user:733

import numpy as np
import random
import json
import sys


class hps(object):
    def __init__(self,
                 filename,
                 num_test=1,
                 learning_rate=0.01,
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

    def reset_train_state(self):
        self.Wx = np.random.randn(10, 10) * 0.5
        self.Wh = np.random.randn(10, 10) * 0.5
        self.item = np.random.randn(8533, self.hps.hidden_size) * 0.5
        self.idx_inps = np.zeros(self.hps.num_unrolling, dtype=np.int)

    def reset_state(self):
        self.h = np.zeros((self.hps.num_unrolling, self.hps.hidden_size))

    def train(self):
        for n in range(self.num_user):
            self.reset_state()
            for t in range(len(self.train_data[n]) - self.hps.num_test):
                sys.stdout.write('Training user: %10d, item: %10d\r' % (n,t))
                sys.stdout.flush()
                self.train_step(
                    self.train_data[n][t] - 1, self.train_data[n][t + 1] - 1, self.neg[n][t + 1])

    def eval(self):
        num_pred = 0
        all_pred = 0
        for n in range(self.num_user):
            self.reset_state()
            for t in range(len(self.train_data[n])):
                pred = self.prediction(self.train_data[n][t])
            for t in range(self.hps.num_test):
                tmp = np.dot(pred, self.item.T)
                sort_list = np.argsort(-tmp)
                for k in range(10):
                    if sort_list[k] == self.data[n][-self.hps.num_test + t]:
                        num_pred += 1
                        break
                all_pred += 1
        return num_pred, all_pred

    def prediction(self, idx_inp):
        tmp = np.dot(self.item[idx_inp], self.Wx)+ np.dot(self.h[0], self.Wh)
        self.h[0] = tmp
        return self.h[0]

    def train_step(self, idx_inp, idx_pos, idx_neg):
        # 存储输入
        self.idx_inps[1:] = self.idx_inps[:-1]
        self.idx_inps[0] = idx_inp
        # 前向
        self.h[1:] = self.h[:-1]
        self.h[0] = self.sigmoid(np.dot(self.item[self.idx_inps[0]], self.Wx) +
                                 np.dot(self.h[0], self.Wh))
        Xuij = self.h[0] * (self.item[idx_pos] - self.item[idx_neg])
        # 后向
        share = self.sigmoid(-Xuij)
        self.item[idx_pos] = self.hps.learning_rate * \
            (share * self.h[0] - self.hps.lamda * self.item[idx_pos])
        self.item[idx_neg] = self.hps.learning_rate * \
            (-share * self.h[0] - self.hps.lamda * self.item[idx_neg])
        for i in range(self.hps.num_unrolling):
            product = 1
            for j in range(i):
                product = self.h[j] * (1 - self.h[j])
            self.Wx = self.hps.learning_rate * \
                (share * product *
                 (self.item[idx_pos] - self.item[idx_neg]) * self.item[self.idx_inps[i]] -
                 self.hps.lamda * self.Wx)
            self.Wh = self.hps.learning_rate * \
                (share * product *
                 (self.item[idx_pos] - self.item[idx_neg]) * self.h[i] -
                 self.hps.lamda * self.Wh)

    def _load_data(self):
        self.data = []
        self.train_data = []
        self.neg = []
        self.num_user = 0
        with open(self.hps.filename, 'r') as f:
            for line in f:
                linedata = json.loads(line)
                if len(linedata) < self.hps.num_test + 1:
                    continue
                self.num_user += 1
                self.data.append([int(i) for i in linedata])
                self.train_data.append([int(i)
                                        for i in linedata][:-self.hps.num_test])
                tmp = []
                for i in linedata:
                    while True:
                        j = random.randint(0, 8532)
                        if j != int(i - 1):
                            break
                    tmp.append(j)
                self.neg.append(tmp)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


R = rnn(hps('user_cart.json'))
for step in range(500):
    R.train()
    print('Step %d done.                        ' % step)
    num_p, all_p = R.eval()
    print(num_p / all_p)
print('Dnoe.')
