import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_running_mean_std import RunningMeanStd
import numpy as np

class CnnPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, hid_size, num_hid_layers, kind='large'):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, hid_size, num_hid_layers, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, kind):
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

        ob_last = tf.concat((obpz, x), axis=-1)

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

        self.state_in = []
        self.state_out = []
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample() # XXX
        self._act = U.function([stochastic, ob_p, ob_f], [ac, self.vpred])

    def act(self, stochastic, ob):
        ob1, ob2 = ob
        ob2 = np.array(ob2)
        ac1, vpred1 = self._act(stochastic, ob1, ob2)
        return ac1[0], vpred1[0]
    
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []


if __name__ == "__main__":
    from env.env_util import make_env
    import rospy
    rospy.init_node("sample")

    env = make_env("PipelineTrack-v1")()

    pol = CnnPolicy("pi", env.observation_space, env.action_space, hid_size=256, num_hid_layers=1)
    ob = env.reset()

    sess = U.single_threaded_session()
    sess.__enter__()
    
    U.initialize()
    
    a, v = pol.act(True, ob)
    print(a)
    print(v)






        



