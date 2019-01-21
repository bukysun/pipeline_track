import time
import random
from env.env_util import make_env 
import cv2
import numpy as np



def main():
    EPISODES = 5
    env = make_env("PipelineTrack-v1", frame_stack=True)()

    steps = 100
    #num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    #print ("Number of States:", num_states)
    print ("Number of Actions:", num_actions)
    print ("Number of Steps per episode:", steps)

    for episode in range(EPISODES):
        time.sleep(0.1)
        state = env.reset()
        frames = np.array(state[1])
        #cv2.imwrite("start.jpg", state[1])
        time.sleep(0.1)
        total_reward = 0
        epi_buffer = list()
        obs = [state]
        for step in range(steps):
            tau1 = random.uniform(-1,1)
            tau2 = random.uniform(-1,1)
            action = np.array([-0.8,-0.0])
            next_state,reward,done,_= env.step(action)
            if done:
                break
            total_reward += reward      
            state = next_state
            obs.append(state)
        ob1, ob2 = zip(*obs)
        ob1 = np.array(ob1)
        ob2 = np.array(ob2)
        print(ob1.shape)
        print(ob2.shape)

        print ('episode: ', episode+1, '  Train Reward:',total_reward)
    env.close()




if __name__ == "__main__":
    import rospy
    import roslaunch
    from env.ros_utils import launch_from_py
    launch = launch_from_py("auv", "/home/uwsim/uwsim_ws/install_isolated/share/RL/launch/basic.launch")
    #rospy.init_node("auv2", anonymous=True)
    #uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    #roslaunch.configure_logging(uuid)
    #launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/uwsim/uwsim_ws/install_isolated/share/RL/launch/basic.launch"])

    launch.start()
    rospy.loginfo("auv started")
    rospy.sleep(20)
    #rospy.init_node("sample")
    main()
    launch.shutdown()

