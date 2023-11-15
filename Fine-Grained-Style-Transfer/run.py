import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
import time
import random
import pickle

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
sess_conf = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))

print(tf.test.is_gpu_available())

import tensorflow as tf
import os

def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.compat.v1.variable_scope(scope or "SimpleLinear"):
        matrix = tf.compat.v1.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.compat.v1.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.compat.v1.matmul(input_, tf.compat.v1.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.compat.v1.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.compat.v1.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sess, sequence_length, num_classes, vocab_size, dp,
            emd_dim, filter_sizes, num_filters, l2_reg_lambda=0.0, dropout_keep_prob = 1):
        # Placeholders for input, output and dropout
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_rate = dropout_keep_prob
        self.dropout_input = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        self.sess = sess
        self.max_sentence_len = sequence_length
        self.dp = dp
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.compat.v1.constant(0.0)

        with tf.compat.v1.variable_scope('TextCNN'):
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.compat.v1.random_uniform([vocab_size, emd_dim], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, emd_dim, 1, num_filter]
                    W = tf.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_input)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.compat.v1.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.compat.v1.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
                self.d_loss = tf.reshape(tf.reduce_mean(self.loss), shape=[1])
                
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
                
        self.params = tf.compat.v1.trainable_variables()
        d_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)
        #self.saver = tf.train.Saver([v for v in tf.trainable_variables() if 'summary_' not in v.name], max_to_keep = 5)
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables(), max_to_keep = 5)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
    def save(self, path, epoch):
        checkpoint_prefix = os.path.join(path, "model")
        self.saver.save(self.sess, checkpoint_prefix, global_step=epoch)
        print('save to %s success' % checkpoint_prefix)
        
    def restore(self, path):
        self.saver.restore(self.sess, path)
        print('restore %s success' % path)
        
    def setup_summary(self):
        train_loss_ = tf.Variable(0., name='summary_train_loss')
        tf.compat.v1.summary.scalar('Train_loss', train_loss_)
        train_acc_ = tf.Variable(0., name='summary_train_acc')
        tf.compat.v1.summary.scalar('Train_Acc', train_acc_)
        
        test_loss_ = tf.Variable(0., name='summary_train_loss')
        tf.compat.v1.summary.scalar('Train_loss', test_loss_)
        test_acc_ = tf.Variable(0., name='summary_test_acc')
        tf.compat.v1.summary.scalar('Test_Acc', test_acc_)
        
        summary_vars = [train_loss_, train_acc_, test_loss_, test_acc_]
        summary_placeholders = [tf.compat.v1.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.compat.v1.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def infer(self, x_str):
        X_ind = [self.dp.w2id[w] for w in x_str.split()]
        X_pad_ind = [X_ind + [self.dp._x_pad] * (self.dp.max_length - len(X_ind))]
        #print(X_pad_ind)
        predict = self.sess.run(self.predictions, 
                {self.input_x: X_pad_ind,
                self.dropout_input: 1.0})[0]
        return predict
        

class TextCNN_DP:
    def __init__(self, X_indices, C_labels, w2id, batch_size, max_length, n_epoch, split_ratio=0.1, test_data=None):
        self.n_epoch = n_epoch
        if test_data == None:
            num_test = int(len(X_indices) * split_ratio)
            r = np.random.permutation(len(X_indices))
            X_indices = np.array(X_indices)[r].tolist()
            C_labels = np.array(C_labels)[r].tolist()
            self.C_train = np.array(C_labels[num_test:])
            self.X_train = np.array(X_indices[num_test:])
            self.C_test = np.array(C_labels[:num_test])
            self.X_test = np.array(X_indices[:num_test])
        else:
            self.X_train, self.C_train, self.X_test, self.C_test = test_data
            self.X_train = np.array(self.X_train, dtype=object)
            self.C_train = np.array(self.C_train, dtype=object)
            self.X_test = np.array(self.X_test, dtype=object)
            self.C_test = np.array(self.C_test, dtype=object)
        self.max_length = max_length
        self.num_batch = int(len(self.X_train) / batch_size)
        self.num_steps = self.num_batch * self.n_epoch
        self.batch_size = BATCH_SIZE
        self.w2id = w2id
        self.id2w = dict(zip(w2id.values(), w2id.keys()))
        self._x_pad = w2id['<PAD>']
        print('Train_data: %d | Test_data: %d | Batch_size: %d | Num_batch: %d | vocab_size: %d' % (len(self.X_train), len(self.X_test), BATCH_SIZE, self.num_batch, len(self.w2id)))
        
    def next_batch(self, X, C):
        r = np.random.permutation(len(X))
        X = X[r]
        C = C[r]
        for i in range(0, len(X) - len(X) % self.batch_size, self.batch_size):
            X_batch = X[i : i + self.batch_size]
            C_batch = C[i : i + self.batch_size]
            padded_X_batch = self.pad_sentence_batch(X_batch, self._x_pad)
            yield (np.array(padded_X_batch),
                   C_batch)
    
    def sample_test_batch(self):
        i = random.randint(0, int(len(self.C_test) / self.batch_size)-2)
        C = self.C_test[i*self.batch_size:(i+1)*self.batch_size]
        padded_X_batch = self.pad_sentence_batch(self.X_test[i*self.batch_size:(i+1)*self.batch_size], self._x_pad)
        return np.array(padded_X_batch), C
    
        
    def pad_sentence_batch(self, sentence_batch, pad_int):
        padded_seqs = []
        seq_lens = []
        sentence_batch = sentence_batch.tolist()
        max_sentence_len = self.max_length
        for sentence in sentence_batch:
            padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
            seq_lens.append(len(sentence))
        return padded_seqs
    
class TextCNN_Util:
    def __init__(self, dp, model, display_freq=3):
        self.display_freq = display_freq
        self.dp = dp
        self.D = model
        
    def train(self, epoch):
        avg_c_loss = 0.0
        avg_acc = 0.0
        tic = time.time()
        X_test_batch, C_test_batch  = self.dp.sample_test_batch()
        for local_step, (X_train_batch, C_train_batch) in enumerate(
            self.dp.next_batch(self.dp.X_train, self.dp.C_train)):
            #print(len(C_train_batch), len(X_train_batch))
            acc, loss, _ = self.D.sess.run([self.D.accuracy, self.D.d_loss, self.D.train_op], 
                {self.D.input_x: X_train_batch, 
                self.D.input_y: C_train_batch, 
                self.D.dropout_input: self.D.dropout_rate})
            avg_c_loss += loss
            avg_acc += acc
            if (local_step % int(self.dp.num_batch / self.display_freq)) == 0:
                val_acc, val_c_loss = self.D.sess.run([self.D.accuracy, self.D.d_loss], 
                                            {self.D.input_x: X_test_batch, 
                                            self.D.input_y: C_test_batch, 
                                            self.D.dropout_input: self.D.dropout_rate})
                print("Epoch %d/%d | Batch %d/%d | Train_loss: %.3f Acc %.3f | Test_loss: %.3f Acc %.3f | Time_cost:%.3f" % 
                      (epoch, self.n_epoch, local_step, self.dp.num_batch, avg_c_loss / (local_step + 1), avg_acc / (local_step + 1), val_c_loss, val_acc, time.time()-tic))
                self.cal()
                tic = time.time()
        return avg_c_loss / (local_step + 1), avg_acc / (local_step + 1)
    
    def test(self):
        avg_c_loss = 0.0
        avg_acc = 0.0
        tic = time.time()
        for local_step, (X_test_batch, C_test_batch) in enumerate(
            self.dp.next_batch(self.dp.X_test, self.dp.C_test)):
            acc, loss = self.D.sess.run([self.D.accuracy, self.D.d_loss], 
                {self.D.input_x: X_test_batch, 
                self.D.input_y: C_test_batch, 
                self.D.dropout_input: 1.0})
            avg_c_loss += loss
            avg_acc += acc
        return avg_c_loss / (local_step + 1), avg_acc / (local_step + 1)
    
    def fit(self, train_dir):
        self.n_epoch = self.dp.n_epoch
        out_dir = train_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print("Writing to %s" % out_dir)
        checkpoint_prefix = os.path.join(out_dir, "model")
        # self.summary_writer = tf.compat.v1.summary.FileWriter(os.path.join(out_dir, 'Summary'), self.D.sess.graph)
        for epoch in range(1, self.n_epoch+1):
            tic = time.time()
            train_c_loss, train_acc = self.train(epoch)
            test_c_loss, test_acc = self.test()
            print("Epoch %d/%d | Train_loss: %.3f Acc %.3f | Test_loss: %.3f Acc %.3f" % 
                  (epoch, self.n_epoch, train_c_loss, train_acc, test_c_loss, test_acc))
            path = self.D.saver.save(self.D.sess, checkpoint_prefix, global_step=epoch)
            print("Saved model checkpoint to %s" % path)
    
    def show(self, sent, id2w):
        return " ".join([id2w.get(idx, u'&') for idx in sent])
    
    def cal(self, n_example=5):
        train_n_example = int(n_example / 2)
        test_n_example = n_example - train_n_example
        for _ in random.sample([t for t in range(len(self.dp.X_test))], test_n_example):
            example = self.show(self.dp.X_test[_], self.dp.id2w)
            o = self.D.infer(example)
            print('Test Input: %s | Output: %d | GroundTruth: %d' % (example, o, np.argmax(self.dp.C_test[_])))
        for _ in random.sample([t for t in range(len(self.dp.X_train))], train_n_example):
            example = self.show(self.dp.X_train[_], self.dp.id2w)
            o = self.D.infer(example)
            print('Train Input: %s | Output: %d | GroundTruth: %d' % (example, o, np.argmax(self.dp.C_train[_]))) 
        print("")
        
# Load Data
import pickle
w2id, id2w = pickle.load(open('AEGS_data/yelp/w2id_id2w.pkl','rb'))
Y_train, C_train = pickle.load(open('AEGS_data/yelp/XC_train.pkl','rb'))
Y_dev, C_dev = pickle.load(open('AEGS_data/yelp/XC_dev.pkl','rb'))
Y_test, C_test = pickle.load(open('AEGS_data/yelp/XC_test.pkl','rb'))
print(C_train[0])

X_train = [x[:-1] for x in Y_train]
X_dev = [x[:-1] for x in Y_dev]
X_test = [x[:-1] for x in Y_test]


# The model

BATCH_SIZE = 256
NUM_EPOCH = 30
train_dir ='Model/YELP/TextCNN/'
MAX_LENGTH = 16

dp = TextCNN_DP(None, None, w2id,  BATCH_SIZE, max_length = MAX_LENGTH, n_epoch=NUM_EPOCH, test_data=(X_train, C_train, X_dev, C_dev))

emb_dim = 128
filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]

g2 = tf.Graph()
sess2 = tf.compat.v1.Session(graph=g2, config=sess_conf) 
with sess2.as_default():
    with sess2.graph.as_default():
        D = TextCNN(sess = sess2, dp = dp, sequence_length=MAX_LENGTH, num_classes=2, vocab_size=len(dp.id2w),
                          emd_dim = emb_dim, filter_sizes = filter_sizes, num_filters=num_filters,
                          l2_reg_lambda=0.2, dropout_keep_prob=0.75)
        D.sess.run(tf.compat.v1.global_variables_initializer())
        
tf.compat.v1.disable_v2_behavior()
util = TextCNN_Util(dp=dp, model=D)
util.fit(train_dir=train_dir)

