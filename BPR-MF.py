import random
import numpy as np
import sys
from scipy.sparse import csr_matrix
from math import exp


class Data(object):
    def __init__(self, train_Path, test_Path):
        '''
        users and items are zero-indexed
        '''
        self.train_Path = train_Path
        self.test_Path = test_Path
        self.num_users = 943
        self.num_items = 1682
        f = open(train_Path)
        train_data = np.array(
            [list(map(int, line.strip().split()[0:2])) for line in f])
        self.train_data = csr_matrix((np.ones(
            (len(train_data))), (train_data[:, 0] - np.ones((len(train_data))),
                                 train_data[:, 1] - np.ones(
                                     (len(train_data))))))

        f = open(test_Path)
        test_data = np.array(
            [list(map(int, line.strip().split()[0:2])) for line in f])
        self.test_data = csr_matrix((np.ones(
            (len(test_data))), (test_data[:, 0] - np.ones((len(test_data))),
                                test_data[:, 1] - np.ones((len(test_data))))))

    def generate_samples(self):
        idxs = list(range(self.train_data.nnz))
        random.shuffle(idxs)
        self.users, self.items = self.train_data.nonzero()
        self.users = self.users[idxs]
        self.items = self.items[idxs]
        self.idx = 0
        for _ in range(self.train_data.nnz):
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.train_data[u].indices)
            self.idx += 1
            yield u, i, j

    def sample_negative_item(self, user_items):
        j = self.random_item()
        while j in user_items:
            j = self.random_item()
        return j

    def creat_loss_samples(self):
        k = 0
        for _ in range(self.test_data.nnz):
            k += 1
            if k % 1000 == 0:
                sys.stdout.write('Creating loss samples:%10d\r' % k)
                sys.stdout.flush()
            u = self.random_user()
            if len(self.test_data[u].indices) > 0:
                i = random.choice(self.test_data[u].indices)
                j = self.sample_negative_item(
                    np.concatenate([
                        self.test_data[u].indices, self.train_data[u].indices
                    ]))
                yield u, i, j

    def random_user(self):
        return random.randint(0, self.num_users - 1)

    def random_item(self):
        return random.randint(0, self.num_items - 1)


class BPRargs(object):
    def __init__(self,
                 learning_rate=0.05,
                 user_regularization=0.002,
                 positive_item_regularization=0.002,
                 negative_item_regularization=0.002):
        self.learning_rate = learning_rate
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization


class BPR(object):
    def __init__(self, f, args):
        self.f = f
        self.learning_rate = args.learning_rate
        self.user_regularization = args.user_regularization
        self.positive_item_regularization = args.positive_item_regularization
        self.negative_item_regularization = args.negative_item_regularization

    def train(self, data, num_iters):
        self.init(data)
        print('initial AUC = {0}'.format(self.loss()))
        for it in range(num_iters):
            k = 0
            for u, i, j in data.generate_samples():
                k += 1
                if k % 1000 == 0:
                    sys.stdout.write('Training:%10d\r' % k)
                    sys.stdout.flush()
                self.update_factors(u, i, j)
            print('iteration {0}: AUC = {1}'.format(it, self.loss()))

    def init(self, data):
        self.data = data
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.W = np.random.random_sample((self.num_users, self.f))
        self.H = np.random.random_sample((self.num_items, self.f))
        self.loss_samples = [t for t in data.creat_loss_samples()]

    def update_factors(self, u, i, j):
        x = np.dot(self.W[u, :], self.H[i, :] - self.H[j, :])
        z = 1.0 / (1.0 + exp(x))

        d = (self.H[i, :] - self.H[j, :]
             ) * z - self.user_regularization * self.W[u, :]
        self.W[u, :] += self.learning_rate * d

        d = self.W[u, :] * z - self.positive_item_regularization * self.H[i, :]
        self.H[i, :] += self.learning_rate * d

        d = -self.W[u, :] * z - self.negative_item_regularization * self.H[j, :]
        self.H[j, :] += self.learning_rate * d

    def loss(self):
        AUC = 0
        k = 0
        E = np.zeros((self.num_users, 2))
        for u, i, j in self.loss_samples:
            k += 1
            if k % 1000 == 0:
                sys.stdout.write('Calculating AUC:%10d\r' % k)
                sys.stdout.flush()
            E[u][0] += 1
            if self.predict(u, i) - self.predict(u, j) > 0:
                E[u][1] += 1
        for e in E:
            if e[0] > 0:
                AUC += e[1] / e[0]
        AUC /= self.num_users
        return AUC

    def predict(self, u, i):
        return np.dot(self.W[u], self.H[i])


if __name__ == '__main__':
    data = Data('train.txt', 'test.txt')
    args = BPRargs()
    model = BPR(10, args)
    model.train(data, 20)
