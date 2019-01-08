import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from auv_uwsim import AuvUwsim

class PipelineTrackEnv(gym.Env):

    def __init__(self):
        self.dynamics = AuvUwsim()

        # action space
        self.action_space = spaces.Box(low = -np.array(np.ones(5)), high = np.array(np.ones(5)))
        self.seed()

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return ([seed])

    def reset(self):
        init_state = [0,1,7.8,0,0,1.27,0, 0.0,0.0,0.0,0.0,0.0] 
        self.state = self.dynamics.reset_sim(init_state)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state, reward = self.dynamics.frame_step(action)
        return self._get_obs(), reward, False, None

    def _get_obs(self):
        return np.array(self.state)



def main():
    import time
    import random
    EPISODES = 1
    env = PipelineTrackEnv()
    steps = 100
    #num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    #print ("Number of States:", num_states)
    print ("Number of Actions:", num_actions)
    print ("Number of Steps per episode:", steps)
    
    for episode in range(EPISODES):
        time.sleep(1)
        state = env.reset()
        time.sleep(1)
        total_reward = 0
        epi_buffer = list()
        for step in range(steps):
            tau1 = random.uniform(-1,1)
            tau2 = random.uniform(-1,1)
            action = np.array([0.8,0.8,0,0,0])
            next_state,reward,done,_= env.step(action)
            total_reward += reward      
            state = next_state       
        print ('episode: ', episode+1, '  Train Reward:',total_reward)




if __name__ == "__main__":
    import rospy
    rospy.init_node("sample")
    main()

