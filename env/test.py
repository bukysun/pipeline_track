


def main():
    import time
    import random
    from env_util import make_env 
    import cv2
    EPISODES = 5
    env = make_env("PipelineTrack-v1")()

    steps = 100
    #num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    #print ("Number of States:", num_states)
    print ("Number of Actions:", num_actions)
    print ("Number of Steps per episode:", steps)

    for episode in range(EPISODES):
        #time.sleep(1)
        state = env.reset()
        frames = np.array(state[1])
        #cv2.imwrite("start.jpg", state[1])
        #time.sleep(1)
        total_reward = 0
        epi_buffer = list()
        for step in range(steps):
            tau1 = random.uniform(-1,1)
            tau2 = random.uniform(-1,1)
            action = np.array([-0.8,-0.0])
            next_state,reward,done,_= env.step(action)
            if done:
                break
            total_reward += reward      
            state = next_state
            frames = np.array(state[1])
            if step == 10:
                for i in range(frames.shape[-1]):
                    cv2.imwrite("sample%d.jpg"%i, frames[:,:,i])

        print ('episode: ', episode+1, '  Train Reward:',total_reward)




if __name__ == "__main__":
    import rospy
    rospy.init_node("sample")
    main()

