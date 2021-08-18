import timeit, socket, os, sys, functools, time
import numpy
import functools
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import SGD
import tmci.plugins
import tmci.checkpoint

class CkptSGD(SGD):
    def __init__(self, logger = None, **config):
        super().__init__(**config)
        self.logger = logger

    def apply_gradients(self, gv, **kwargs):
        return super().apply_gradients(gv, **{ **kwargs, 'ckpt': self.logger })

    def get_config(self):
        return { 'logger': self.logger, **super().get_config() }

class BatchMonitor(Callback):
    def __init__(self, rank, size, comm, standby):
        self.THRESHOLD = 10
        self.counter = 0
        self.rank = rank
        self.size = size
        self.ckpt_no = 0
        self.comm = comm
        self.standby = standby
        self.active = tf.Variable(False)
        tmci.plugins.load(os.environ.get('TMCI_PLUGIN_LIB') + '/libsmart.so')
        self.save_tensors = functools.partial(tmci.checkpoint.save_tensors, 'smart')
        self.tensors = []

        os.makedirs("ckpt-logs", exist_ok = True)
        self.fn = open(os.path.join("ckpt-logs", "batch-" + socket.gethostname() + "-" + str(self.rank) + ".log"), 'w')
        self.ts = timeit.default_timer()
        self.batch_ts = self.ts
        self.log("begin batch monitoring, assuming role: %d" % (rank % 2))

    def start_standby(self):
        self.log("waiting for weights at epoch %d" % self.counter)
        t = timeit.default_timer()
        tmci.checkpoint.load_tensors('smart', str(len(self.tensors)), self.tensors)
        self.log("weights received at epoch %d" % self.counter, t)
        self.counter = self.THRESHOLD

    def on_batch_begin(self, batch, logs):
        if self.counter == 1 and self.rank % 2 == 1:
            self.start_standby()
        self.counter += 1
        self.log("starting batch %d" % self.counter, self.batch_ts)
        self.batch_ts = timeit.default_timer()
        if self.counter == self.THRESHOLD:
            t = timeit.default_timer()
            tf.keras.backend.set_value(self.active, True)
            self.log("started checkpoint at epoch %d" % self.counter, t)

    def on_batch_end(self, batch, logs):
        if self.counter == self.THRESHOLD:
            t = timeit.default_timer()
            tf.keras.backend.set_value(self.active, False)
            self.log("finalized checkpoint at epoch %d" % self.counter, t)
        self.ckpt_no += 1

    def log(self, line, ref=0.0):
        now = timeit.default_timer()
        if ref == 0.0:
            print("[%.3f] Rank %d: %s" % (now - self.ts, self.rank, line), file = self.fn)
        else:
            print("[%.3f] [duration: %.3f] Rank %d: %s" % (now - self.ts, now - ref, self.rank, line), file = self.fn)
        self.fn.flush()
