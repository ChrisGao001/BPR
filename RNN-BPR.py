import sys
import json
import numpy as np
import tensorflow as tf


num_nodes = 64
start_learning_rate = 10.0
num_steps = 100
summary_frequency = 200
recall_frequency = 10

################ 函数 ##################
def recall(y,y_pred,n):
    if np.shape(y)!=np.shape(y_pred):
        raise ValueError()
    tmp = sorted(y_pred[0])[-n:]
    for i in range(len(y[0])):
        if y[0][i]==1 and y_pred[0][i] in tmp:
            return 1
    return 0
################ 输入 ##################


class Input(object):
    def __init__(self, filename='./user_cart.json', batch_size=10, num_unrollings=10, num_test=1):
        self.batch_size = batch_size
        self.num_unrollings = num_unrollings
        if num_test == 0:
            raise ValueError('num_test must > 0')
        else:
            self.num_test = num_test
        self._load_file(filename)
        self._init_data()

    def _load_file(self, filename):
        # 载入数据文件
        self._data = []
        with open(filename, 'r') as f:
            for line in f:
                if len(line) > 3:
                    # 如果这行非空
                    self._data.append(json.loads(line))
        self.num_item = 8533
        self.num_user = len(self._data)  # 734
        counter = 0
        for item in self._data:
            counter+=len(item)
        print(counter)

    def _init_data(self):
        # 初始化数据
        self._buffer = [self._data[i % self.num_user]
                        for i in range(self.batch_size)]
        self._buffer_pointer = self.batch_size % self.num_user
        self._test_pointer = 0

    def reset_test_pointer(self):
        self._test_pointer = 0

    def next_test_batches(self):
        while len(self._data[self._test_pointer]) < self.num_test:
            self._test_pointer += 1
        t_item = []
        l_item = []
        for j in range(len(self._data[self._test_pointer]) - self.num_test):
            tmp = np.zeros((1, self.num_item))
            tmp[0, int(self._data[self._test_pointer][j]) - 1] = 1
            t_item.append(tmp)
        for j in range(self.num_test):
            tmp = np.zeros((1, self.num_item))
            tmp[0, int(self._data[self._test_pointer]
                       [-self.num_test + j]) - 1] = 1
            l_item.append(tmp)
        self._test_pointer += 1
        if not self._test_pointer < self.num_user:
            return t_item, l_item, True
        else:
            return t_item, l_item, False

    def next_train_batches(self):
        # shape = [num_unrolling, types, batch_size, num_item]
        return [self._next_train_batch() for _ in range(self.num_unrollings)]

    def _next_train_batch(self):
        self._check_buffer()
        # shape = [batch_size, num_item]
        inputs = np.zeros((self.batch_size, self.num_item))
        for i in range(self.batch_size):
            inputs[i, int(self._buffer[i][0]) - 1] = 1
            self._buffer[i] = self._buffer[i][1:]
        # shape = [batch_size, num_item]
        labels = np.zeros((self.batch_size, self.num_item))
        for i in range(self.batch_size):
            labels[i, int(self._buffer[i][0]) - 1] = 1
        # shape = [batch_size, num_item]
        negsam = np.zeros((self.batch_size, self.num_item))
        for i in range(self.batch_size):
            while True:
                k = np.random.randint(self.num_item)
                if k + 1 not in self._buffer[:][0]:
                    negsam[i, k] = 1
                    break
        return inputs, labels, negsam

    def _check_buffer(self):
        for i in range(self.batch_size):
            while len(self._buffer[i]) < self.num_test + 2:
                self._buffer[i] = self._data[self._buffer_pointer]
                self._buffer_pointer = (
                    self._buffer_pointer + 1) % self.num_user

################ 测试 ##################


def test():
    ip = Input()
    train_batches = ip.next_train_batches()
    tmp_t = []
    for unrolling in train_batches:
        tmp_u = []
        for types in unrolling:
            tmp_ty = []
            for item in types:
                for i in range(ip.num_item):
                    if item[i] == 1:
                        tmp_ty.append(i + 1)
            tmp_u.append(tmp_ty)
        tmp_t.append(tmp_u)
    print('next_train_batches():example')
    print('inputs:')
    print(tmp_t[0][0])
    print('labels:')
    print(tmp_t[0][1])
    print('negsam:')
    print(tmp_t[0][2])
    print('all_test():example')
    nxt, y, f = ip.next_test_batches()
    tmp_at = []
    for item in nxt:
        for i in range(ip.num_item):
            if item[0, i] == 1:
                tmp_at.append(i + 1)
    print(tmp_at)
    tmp_y = []
    for item in y:
        for i in range(ip.num_item):
            if item[0, i] == 1:
                tmp_y.append(i + 1)
    print(tmp_y)
    print(f)
    del ip
# test()
################ 构建 ##################


ip = Input()
graph = tf.Graph()
print('building graph')
with tf.device('/gpu:0'):
    with graph.as_default():
    
        # 输入门
        iW = tf.Variable(tf.truncated_normal(
            [ip.num_item, num_nodes], -0.1, 0.1))
        iV = tf.Variable(tf.truncated_normal(
            [num_nodes, num_nodes], -0.1, 0.1))
        ib = tf.Variable(tf.zeros([1, num_nodes]))
        # 遗忘门
        fW = tf.Variable(tf.truncated_normal(
            [ip.num_item, num_nodes], -0.1, 0.1))
        fV = tf.Variable(tf.truncated_normal(
            [num_nodes, num_nodes], -0.1, 0.1))
        fb = tf.Variable(tf.zeros([1, num_nodes]))
        # cell更新
        cW = tf.Variable(tf.truncated_normal(
            [ip.num_item, num_nodes], -0.1, 0.1))
        cV = tf.Variable(tf.truncated_normal(
            [num_nodes, num_nodes], -0.1, 0.1))
        cb = tf.Variable(tf.zeros([1, num_nodes]))
        # 输出门
        oW = tf.Variable(tf.truncated_normal(
            [ip.num_item, num_nodes], -0.1, 0.1))
        oV = tf.Variable(tf.truncated_normal(
            [num_nodes, num_nodes], -0.1, 0.1))
        ob = tf.Variable(tf.zeros([1, num_nodes]))
        # 输出
        saved_h = tf.Variable(tf.zeros([ip.batch_size, num_nodes]))
        # 状态
        saved_c = tf.Variable(tf.zeros([ip.batch_size, num_nodes]))
        # 分类器
        W = tf.Variable(tf.truncated_normal(
            [num_nodes, ip.num_item], -0.1, 0.1))
        b = tf.Variable(tf.zeros([1, ip.num_item]))

        def LSTM_cell(i, h, c):
            input_gate = tf.sigmoid(tf.matmul(i, iW) + tf.matmul(h, iV) + ib)
            forget_gate = tf.sigmoid(tf.matmul(i, fW) + tf.matmul(h, fV) + fb)
            output_gate = tf.sigmoid(tf.matmul(i, oW) + tf.matmul(h, oV) + ob)
            update = tf.tanh(tf.matmul(i, cW) + tf.matmul(h, cV) + cb)
            c = forget_gate * c + input_gate * update
            h = output_gate * tf.tanh(c)
            return h, c

        train_inputs = [tf.placeholder(tf.float32, shape=[
            ip.batch_size, ip.num_item]) for _ in range(ip.num_unrollings)]
        train_labels = [tf.placeholder(tf.float32, shape=[
            ip.batch_size, ip.num_item]) for _ in range(ip.num_unrollings)]
        train_negsam = [tf.placeholder(tf.float32, shape=[
            ip.batch_size, ip.num_item]) for _ in range(ip.num_unrollings)]

        hs = []
        h = saved_h  # shape=[batch_size, num_nodes]
        c = saved_c  # shape=[batch_size, num_nodes]
        for i in train_inputs:
            h, c = LSTM_cell(i, h, c)
            hs.append(h)  # shape=[num_unrollings, batch_size, num_nodes]

        with tf.control_dependencies([saved_c.assign(c), saved_h.assign(h)]):
            # tf.concat(hs,0).shape = [batch_size * num_unrollings, num_nodes]
            # W.shape = [num_nodes, num_item]
            # b.shape = [1, num_item]
            # logits.shape = [batch_size * num_unrollings, num_item]
            logits = tf.matmul(tf.concat(hs, 0), W) + b
            # logits = tf.nn.xw_plus_b(tf.concat(hs, 0), W, b)

        rsi = tf.reduce_sum(tf.concat(train_labels, 0) * logits, 1)
        rsj = tf.reduce_sum(tf.concat(train_negsam, 0) * logits, 1)

        loss = tf.reduce_mean(-tf.log(tf.sigmoid(rsi - rsj)))

        train_prediction = tf.nn.softmax(logits)
        # 优化器
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            start_learning_rate, global_step, 5000, 0.1, staircase=True)  # 调整学习率
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)  # 新建优化器
        gradients, vriables = zip(*optimizer.compute_gradients(loss))  # 计算梯度
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)  # 修剪梯度防止梯度弥散和爆炸
        optimizer = optimizer.apply_gradients(
            zip(gradients, vriables), global_step=global_step)  # 应用梯度

        sample_input = tf.placeholder(tf.float32, shape=[1, ip.num_item])
        saved_sample_h = tf.Variable(tf.zeros([1, num_nodes]))
        saved_sample_c = tf.Variable(tf.zeros([1, num_nodes]))
        reset_sample_hc = tf.group(saved_sample_h.assign(
            tf.zeros([1, num_nodes])), saved_sample_c.assign(tf.zeros([1, num_nodes])))
        sample_h, sample_c = LSTM_cell(
            sample_input, saved_sample_h, saved_sample_c)
        with tf.control_dependencies([saved_sample_h.assign(sample_h),
                                        saved_sample_c.assign(sample_c)]):
            sample_prediction = tf.nn.softmax(tf.matmul(sample_h, W) + b)


    ################ 训练 ##################


    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        mean_loss = 0
        ip = Input()
        step=0
        for step in range(num_steps*summary_frequency*recall_frequency):
            step +=1
            sys.stdout.write('Training step: %10d\r' % step)
            sys.stdout.flush()
            batches = ip.next_train_batches()
            feed_dict = dict()
            for i in range(ip.num_unrollings):
                feed_dict[train_inputs[i]] = batches[i][0]
                feed_dict[train_labels[i]] = batches[i][1]
                feed_dict[train_negsam[i]] = batches[i][2]
            _, l, predictions, lr = session.run(
                [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
            mean_loss += l
            if step % summary_frequency == 0:
                if step > 0:
                    mean_loss = mean_loss / summary_frequency
                    print('Average loss at step %d: %f learning rate: %f' %
                          (step, mean_loss, lr))
                    mean_loss = 0
                    if step % (summary_frequency * recall_frequency) == 0:
                        # 重置测试指针
                        ip.reset_test_pointer()
                        r = 0
                        num = 0
                        counter1 = 0
                        while True:
                            counter1 += 1

                            # 获取测试数据
                            samples, y, end = ip.next_test_batches()
                            # 重置模型参数
                            reset_sample_hc.run()

                            # 训练
                            for item in samples:
                                sys.stdout.write(
                                    'Calcing recall@10 user: %10d\r' % counter1)
                                sys.stdout.flush()
                                feed = item
                                prediction = sample_prediction.eval(
                                    {sample_input: feed})

                            # 预测
                            y_pred = []
                            counter2 = 0
                            while True:
                                counter2 += 1
                                r += recall(y[counter2 - 1],prediction,10)
                                num += 1
                                if not counter2 < ip.num_test:
                                    break
                                feed = prediction
                                prediction = sample_prediction.eval(
                                    {sample_input: feed})
                            # 判定是否结束
                            if end:
                                break
                        r /= num
                        print('Recall 10 at step %d: %f' % (step, r))
