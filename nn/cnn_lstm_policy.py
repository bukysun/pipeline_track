import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_running_mean_std import RunningMeanStd
import numpy as np


class CnnLSTMPolicy(object):
    def __init__(self, name, ob_space, ac_space, hid_size, num_hid_layers, lstm_hid_size=64, kind='large'):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, hid_size, num_hid_layers, lstm_hid_size, kind)
            self.scope = tf.get_variable_scope().name
            self.recurrent = True

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, lstm_hid_size, kind):
        assert isinstance(ob_space, tuple)
        print("This is lstm policy for all.")

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None
        
        ob_p = U.get_placeholder(name="ob_physics", dtype=tf.float32, shape=[sequence_length] + list(ob_space[0].shape))
        ob_f= U.get_placeholder(name="ob_frames", dtype=tf.float32, shape=[sequence_length]+list(ob_space[1].shape))

        #process ob_p
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape = ob_space[0].shape)
        obpz = tf.clip_by_value((ob_p - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            

        #process ob_f
        x = ob_f / 255.0

        if kind == 'small': # from A3C paper
            x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        elif kind == 'large': # Nature DQN
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        else:
            raise NotImplementedError
        
        ob_last = tf.concat((obpz, x), axis = -1)

        # lstm layer for memmory
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_hid_size, state_is_tuple=True, name = "rnn")
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.state_init = (c_init, h_init)
        c_in = U.get_placeholder(name="state_c", dtype=tf.float32,shape=(None, lstm_cell.state_size.c))
        h_in = U.get_placeholder(name="state_h", dtype=tf.float32,shape=(None, lstm_cell.state_size.h))
        self.state_in = (c_in, h_in)
        
        state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_states = lstm_cell(ob_last, state_in)
        lstm_c, lstm_h = lstm_states
        self.state_out = (lstm_c, lstm_h)

        rnn_out = tf.reshape(lstm_outputs, (-1, lstm_hid_size))

        # value network
        with tf.variable_scope("vf"):
            last_out = rnn_out
            for i in range(num_hid_layers):
                last_out = tf.nn.relu(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]
 
        with tf.variable_scope("pol"):
            last_out = rnn_out
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            logits = tf.layers.dense(last_out, pdtype.param_shape()[0], name='logits', kernel_initializer=U.normc_initializer(0.01))
            self.pd = pdtype.pdfromflat(logits)

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample()
        self._act = U.function([stochastic, ob_p, ob_f, c_in, h_in], [ac, self.vpred, lstm_c, lstm_h])

    def act(self, stochastic, ob, state_in):
        ob1, ob2 = ob
        ob2 = np.array(ob2)
        sc, sh = state_in
        ac1, vpred1, sco, sho = self._act(stochastic, ob1, ob2, sc, sh)
        state_out = (sco, sho)
        return ac1[0], vpred1[0], state_out


    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return self.state_init


class CnnPhyLSTMPolicy(CnnLSTMPolicy):
    
    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, lstm_hid_size, kind):
        print("This is lstm policy for only physics.")
        assert isinstance(ob_space, tuple)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None
        
        ob_p = U.get_placeholder(name="ob_physics", dtype=tf.float32, shape=[sequence_length] + list(ob_space[0].shape))
        ob_f= U.get_placeholder(name="ob_frames", dtype=tf.float32, shape=[sequence_length]+list(ob_space[1].shape))

        #process ob_p
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape = ob_space[0].shape)
        obpz = tf.clip_by_value((ob_p - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            

        #process ob_f
        x = ob_f / 255.0

        if kind == 'small': # from A3C paper
            x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        elif kind == 'large': # Nature DQN
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        else:
            raise NotImplementedError

        # lstm layer for memmory
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_hid_size, state_is_tuple=True, name = "rnn")
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.state_init = (c_init, h_init)
        c_in = U.get_placeholder(name="state_c", dtype=tf.float32,shape=(None, lstm_cell.state_size.c))
        h_in = U.get_placeholder(name="state_h", dtype=tf.float32,shape=(None, lstm_cell.state_size.h))
        self.state_in = (c_in, h_in)

        state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_states = lstm_cell(obpz, state_in)
        lstm_c, lstm_h = lstm_states
        self.state_out = (lstm_c, lstm_h)

        rnn_out = tf.reshape(lstm_outputs, (-1, lstm_hid_size))
        
        # conjugate sensor and physics
        ob_last = tf.concat((rnn_out, x), axis = -1)

        # value network
        with tf.variable_scope("vf"):
            last_out = ob_last
            for i in range(num_hid_layers):
                last_out = tf.nn.relu(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]
 
        with tf.variable_scope("pol"):
            last_out = ob_last
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            logits = tf.layers.dense(last_out, pdtype.param_shape()[0], name='logits', kernel_initializer=U.normc_initializer(0.01))
            self.pd = pdtype.pdfromflat(logits)

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample()
        self._act = U.function([stochastic, ob_p, ob_f, c_in, h_in], [ac, self.vpred, lstm_c, lstm_h])

class CnnSenLSTMPolicy(CnnLSTMPolicy):
    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, lstm_hid_size, kind):
        print("This is lstm policy for only sensors.")
        assert isinstance(ob_space, tuple)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None
        
        ob_p = U.get_placeholder(name="ob_physics", dtype=tf.float32, shape=[sequence_length] + list(ob_space[0].shape))
        ob_f= U.get_placeholder(name="ob_frames", dtype=tf.float32, shape=[sequence_length]+list(ob_space[1].shape))

        #process ob_p
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape = ob_space[0].shape)
        obpz = tf.clip_by_value((ob_p - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            

        #process ob_f
        x = ob_f / 255.0

        if kind == 'small': # from A3C paper
            x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        elif kind == 'large': # Nature DQN
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        else:
            raise NotImplementedError

        # lstm layer for memmory
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_hid_size, state_is_tuple=True, name = "rnn")
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.state_init = (c_init, h_init)
        c_in = U.get_placeholder(name="state_c", dtype=tf.float32,shape=(None, lstm_cell.state_size.c))
        h_in = U.get_placeholder(name="state_h", dtype=tf.float32,shape=(None, lstm_cell.state_size.h))
        self.state_in = (c_in, h_in)

        state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_states = lstm_cell(x, state_in)
        lstm_c, lstm_h = lstm_states
        self.state_out = (lstm_c, lstm_h)

        rnn_out = tf.reshape(lstm_outputs, (-1, lstm_hid_size))
        
        # conjugate sensor and physics
        ob_last = tf.concat((rnn_out, obpz), axis = -1)

        # value network
        with tf.variable_scope("vf"):
            last_out = ob_last
            for i in range(num_hid_layers):
                last_out = tf.nn.relu(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]
 
        with tf.variable_scope("pol"):
            last_out = ob_last
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            logits = tf.layers.dense(last_out, pdtype.param_shape()[0], name='logits', kernel_initializer=U.normc_initializer(0.01))
            self.pd = pdtype.pdfromflat(logits)

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample()
        self._act = U.function([stochastic, ob_p, ob_f, c_in, h_in], [ac, self.vpred, lstm_c, lstm_h])
   




