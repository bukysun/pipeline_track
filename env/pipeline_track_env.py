import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from auv_uwsim import AuvUwsim

class PipelineTrackEnv(gym.Env):

    def __init__(self):
        self.dynamics = AuvUwsim()

        # observation space is a tuple
        ob_space1 = spaces.Box(low=-np.inf, high=np.inf, shape = (5,), dtype=np.float)
        ob_space2 = spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8) 
        self.observation_space = (ob_space1, ob_space2)

        # action space
        self.action_space = spaces.Box(low = -np.array(np.ones(2)), high = np.array(np.ones(2)))
        self.seed()

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return ([seed])

    def reset(self):
        self.state, self.camera = self.dynamics.reset_sim()
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state, self.camera, reward, done, info = self.dynamics.frame_step(action)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert self.camera != [], "get null img"
        x, y, psi, u, v, r = self.state
        ret_state = [np.cos(psi), np.sin(psi), u, v, r]
        return ret_state, self.camera




