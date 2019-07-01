#!/usr/bin/python3

import sys, time, re, math
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from statsmodels.tsa.arima_model import ARIMA

class parser:
    def __init__(self):
        self.mem_def = re.compile('\[([0-9]*\.?[0-9]*)\] node ([0-9]+): n_msg = ([0-9]+); bytes = ([0-9]+); sys_bytes = ([0-9]+)')
        self.time = []
        self.nbytes = []
        self.sbytes = []

    def parse(self, line):
        m = self.mem_def.match(line)
        if m:
            self.time.append(m.group(1))
            self.nbytes.append(m.group(4))
            self.sbytes.append(m.group(5))

def gen_first_X_Y(data, seq_length):
    batch_x = []
    batch_y = []
	
    y_end = 2*seq_length	
    x = data[:seq_length]
    y = data[seq_length:y_end]

    x_ = np.array([x])
    y_ = np.array([y])
    x_, y_ = x_.T, y_.T
    
    batch_x.append(x_)
    batch_y.append(y_)
    
    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    
    return batch_x, batch_y

def gen_X_Y(data, seq_length, data_gen_stride):
    batch_x = []
    batch_y = []
   
    i = seq_length*2
    while i <= len(data):
        x_start = i - (2*seq_length)
        x_end = x_start + seq_length
        
        x = data[x_start:x_end]
        y = data[x_end:i]
        
        x_ = np.array([x])
        y_ = np.array([y])
        x_, y_ = x_.T, y_.T
       
        batch_x.append(x_)
        batch_y.append(y_)
       
        i = i + data_gen_stride
       
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    
    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))

    return batch_x, batch_y

def gen_pred_X_Y(data, seq_length, start_idx):
    batch_x = []
    batch_y = []
    
    x_start = start_idx - seq_length    
    y_end = start_idx + seq_length
    
    x = data[x_start:start_idx]
    y = data[start_idx:y_end]
    
    x_ = np.array([x])
    y_ = np.array([y])
    x_, y_ = x_.T, y_.T
    
    batch_x.append(x_)
    batch_y.append(y_)
    
    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))

    return batch_x, batch_y

# Parsing the log file
if len(sys.argv) < 3:
    print("Usage: %s <log_file> <case>" % sys.argv[0])
    sys.exit(1)

print("Parsing %s" % sys.argv[1])
log_name = sys.argv[1];
log_file = open(log_name, "r")
case = sys.argv[2]

p = parser()
for line in log_file:
    p.parse(line)

log_file.close()

nbytes_list = list(map(lambda x: float(x) / 2**20, p.nbytes))
if case == "amg":
    print("AMG: Processed Infiniband data, can be used directly")
    sbytes_list = list(map(lambda x: float(x) / 2**20, p.sbytes)) # AMR logs (did multiply by 4 during Infiniband monitoring)
    seq_length = 20
elif case == "hacc":
    print("HACC: RAW Infiniband data, needs to be multiplied by 4")
    sbytes_list = list(map(lambda x: 4 * float(x) / 2**20, p.sbytes)) # HACC logs (did not multiply by 4 during Infiniband monitoring)
    seq_length = 60
else:
    raise Exception("Unsupported application, needs to be hacc or amg")

total_data_len = len(nbytes_list)
init_data_size = 2 * seq_length   # The size to have the first training data
print("Total data size = %d, init_data_size = %d" % (total_data_len, init_data_size))

data_gen_stride = 1
history_size = 5 * seq_length   # history window to be kept for training every time 
iteration = history_size

# parameters of seq2seq RNN model 
output_dim = input_dim = 1
hidden_dim = 40  # Count of hidden neurons in the recurrent units. 
layers_stacked_count = 3  # Number of stacked recurrent cells, on the neural depth axis. 
w_seed = 15

learning_rate = 0.005  # Small lr helps not to diverge during training. 
lr_decay = 0.5 # default: 0.9 . Simulated annealing.
momentum = 0.6 # default: 0.0 . Momentum technique in weights update
lambda_l2_reg = 0.002  # L2 regularization of weights - avoids overfitting

# Backward compatibility for TensorFlow's version 0.12: 
try:
    tf.nn.seq2seq = tf.contrib.legacy_seq2seq
    tf.nn.rnn_cell = tf.contrib.rnn
    tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
    print("TensorFlow's version : 1.0 (or more)")
except: 
    print("TensorFlow's version : 0.12")

tf.reset_default_graph()
sess = tf.InteractiveSession()

with tf.variable_scope('Seq2seq'):
    tf.set_random_seed(w_seed)

    # Encoder: inputs
    enc_inp = [
        tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
           for t in range(seq_length)
    ]

    # Decoder: expected outputs
    expected_sparse_output = [
        tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_sparse_output_".format(t))
          for t in range(seq_length)
    ]
    
    # Give a "GO" token to the decoder. 
    # Note: we might want to fill the encoder with zeros or its own feedback rather than with "+ enc_inp[:-1]"
    dec_inp = [ tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO") ] + enc_inp[:-1] #XXX

    # Create a `layers_stacked_count` of stacked RNNs (GRU cells here). 
    cells = []
    for i in range(layers_stacked_count):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
            # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    
    # Here, the encoder and the decoder uses the same cell, HOWEVER,
    # the weights aren't shared among the encoder and decoder, we have two
    # sets of weights created under the hood according to that function's def. 
    dec_outputs, dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(
        enc_inp, 
        dec_inp, 
        cell
    )
    
    # For reshaping the output dimensions of the seq2seq RNN:
    #w_var = tf.get_variable(name = 'w', shape = [output_dim], initializer = initializer) 
    w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
    #b_var = tf.get_variable(name = 'b', shape = [output_dim], initializer = initializer) 
    b_out = tf.Variable(tf.random_normal([output_dim]))


    # Final outputs: with linear rescaling for enabling possibly large and unrestricted output values.
    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
    reshaped_outputs = [output_scale_factor*(tf.matmul(i, w_out) + b_out) for i in dec_outputs]

# Training loss and optimizer

with tf.variable_scope('Loss'):
    # L2 loss
    output_loss = 0
    for _y, _Y in zip(reshaped_outputs, expected_sparse_output):
        output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))
                
    # L2 regularization (to avoid overfitting and to have a  better generalization capacity)
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
            
    loss = output_loss + lambda_l2_reg * reg_loss

with tf.variable_scope('Optimizer'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=lr_decay, momentum=momentum)
    train_op = optimizer.minimize(loss)

def train_first(data, seq_length, X, Y):
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


# Online training and learning
history = []
pred = []
dnn_training = []

sess.run(tf.global_variables_initializer())

first_train_X, first_train_Y = gen_first_X_Y(nbytes_list, seq_length)
first_train_start = time.time()
for t in range(iteration):
        first_loss = train_first(nbytes_list, seq_length, first_train_X, first_train_Y)

first_train_end = time.time()
first_train_time = first_train_end - first_train_start
print("First train, iter = %d, size = 1, loss = %f, training time = %f" % (iteration, first_loss, first_train_time))

first_test_X, first_test_Y = gen_pred_X_Y(nbytes_list, seq_length, 2*seq_length)
feed_dict = {enc_inp[t]: first_test_X[t] for t in range(seq_length)}
outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])

history = nbytes_list[0:init_data_size]

#iteration = iter_stride
#t = init_data_size
loss_prev = first_loss

begin = init_data_size + seq_length
t = begin
total_train_time = 0
while t < total_data_len:
        win_start = t-seq_length
        for i in range(seq_length):
                history.append(nbytes_list[win_start+i])
        extra = 0
        if history_size > len(history):
                extra = history_size - len(history)
        while len(history) > history_size:
                history.pop(0)

        X, Y = gen_X_Y(history, seq_length, data_gen_stride)
        time_start = time.time()
        prev_diff = [10000000000.0 for i in range(5)]
        loss_p = prev_diff[0] - 1
        loss_t = 0.0
        iteration = 0
        while abs(loss_t - loss_p) < max(prev_diff) or iteration < extra:
                prev_diff.append(abs(loss_p - loss_t))
                prev_diff.pop(0)
                loss_p = loss_t
                feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
                feed_dict.update({expected_sparse_output[t]: Y[t] for t in range(len(expected_sparse_output))})
                _, loss_t = sess.run([train_op, loss], feed_dict)
                iteration += 1

        time_end = time.time()
        time_elapsed = time_end - time_start        
        print("Training, iter = %d, loss = %f, training time = %f" % (iteration, loss_t, time_elapsed))
        dnn_training.append(time_elapsed)

        test_X, test_Y = gen_pred_X_Y(nbytes_list, seq_length, t)

        pred_dict = {enc_inp[t]: test_X[t] for t in range(seq_length)}
        test_outputs = np.array(sess.run([reshaped_outputs], pred_dict)[0])
        
        reminder = min(seq_length, total_data_len - t)
        for r in range(reminder):
                pred.append(max(0, test_outputs[r, 0, 0]))
        t = t + seq_length

X_len = len(pred)
real = sbytes_list[begin : begin + X_len]

mse = mean_squared_error(real, pred)
print("DNN MSE: %f" % mse)
dtw, _ = fastdtw(real, pred, dist=euclidean)
print("DNN FastDTW: %.2f" % dtw)

# ARIMA based prediction
t_begin = time.time()
history = nbytes_list[0:init_data_size]
arima = []
arima_training = []
for t in range(init_data_size, len(nbytes_list)):
    model = ARIMA(history, order=(3,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(seq_length)[0]
    yhat = max(0, output[seq_length - 1])
    arima.append(yhat)
    if len(history) > history_size:
       history.pop(0)
    history.append(nbytes_list[t])
    if t % seq_length == 0:
        t_end = time.time()
        t_elapsed = t_end - t_begin
        print("Timestep %d, time elapsed: %.2f" % (t, t_elapsed))
        arima_training.append(t_elapsed)
        t_begin = t_end

arima = arima[0:X_len]
mse = mean_squared_error(real, arima)
print("ARIMA MSE: %f" % mse)
dtw, _ = fastdtw(real, arima, dist=euclidean)
print("ARIMA FastDTW: %.2f" % dtw)

mpi_mon = nbytes_list[begin:begin + X_len]
mse = mean_squared_error(real, mpi_mon)
print("MPI-Mon MSE: %f" % mse)
dtw, _ = fastdtw(real, mpi_mon, dist=euclidean)
print("MPI-Mon FastDTW: %.2f" % dtw)

plt.figure(figsize=(5, 4))
plt.xlabel('Time (s)')
plt.ylabel('Size (MB)')
plt.plot(real, "b-", label = "System level")
plt.plot(mpi_mon, "r-", label = "MPI level")
plt.legend(loc='best')
plt.title("Monitoring accuracy")
plt.savefig(case + '-mpi')

plt.figure(figsize=(5, 4))
plt.xlabel('Time (s)')
plt.ylabel('Size (MB)')
plt.plot(real, "b-", label = "Real")
plt.plot(pred, "r-", label = "Proposal")
plt.plot(arima, "g-", label = "ARIMA")
plt.legend(loc='best')
plt.title("Prediction accuracy")
plt.savefig(case + '-pred')

plt.figure(figsize=(5, 4))
plt.xlabel('Sequence number')
plt.ylabel('Time (s)')
plt.plot(dnn_training, "r-", label = "Proposal")
plt.plot(arima_training[1:], "g-", label = "ARIMA")
plt.legend(loc='best')
plt.title("Compute overhead")
plt.savefig(case + '-time')
