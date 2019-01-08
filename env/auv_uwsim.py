import numpy as np
import math
import rospy
import copy
import cv2
from cv_bridge import CvBridge, CvBridgeError

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int8

import ros_utils as sensors

from env_utils import cenline_extract, get_reward 


class AuvUwsim(object):
    
    def __init__(self):
        #Initialize sensors
        self.IG = sensors.imageGrabber()
        self.DI = sensors.depthImage()
        self.State_p = sensors.GetPose()
        self.State_v = sensors.GetVelocity()

        # Publisher
        self.Thruster_pub = rospy.Publisher("/vehicle1/thrusters_input",Float64MultiArray ,queue_size=1)
        self.reset_pub = rospy.Publisher("/vehicle1/resetdata",Odometry ,queue_size=1)
        self.pause_pub = rospy.Publisher("/pause",Int8,queue_size=1)

    def reset_sim(self, init_state):
        self.state = init_state
        self.step = 0
        self.last_rew = None
        #set initial parameter
        msg = Odometry()

        x, y, z, phi, theta, psi, u, v, w, p, q, r = self.state

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z # 4.5
        msg.pose.pose.orientation.w = np.cos(phi/2)*np.cos(psi/2)*np.cos(theta/2)+np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
        msg.pose.pose.orientation.x = np.sin(phi/2)*np.cos(psi/2)*np.cos(theta/2)-np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2) 
        msg.pose.pose.orientation.y = np.cos(phi/2)*np.cos(psi/2)*np.sin(theta/2)+np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
        msg.pose.pose.orientation.z = np.cos(phi/2)*np.sin(psi/2)*np.cos(theta/2)-np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)

        msg.twist.twist.linear.x = u
        msg.twist.twist.linear.y = v
        msg.twist.twist.linear.z = w
        msg.twist.twist.angular.x = p
        msg.twist.twist.angular.y = q
        msg.twist.twist.angular.z = r

        self.reset_pub.publish(msg)

        #publish reset_flag
        flag = Int8()
        flag.data = 1
        self.pause_pub.publish(flag)
        return self.state

    def frame_step(self, action):
        # show image
        #cv2.imshow("camera", self.IG.cv_image)
        #cv2.imshow("depthImage", self.DI.depth_image)
        #res = cenline_extract(img, "points")
        #if res is not None:
        #    x1, y1, x2, y2 = res
        #    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #else:
        #    cv2.imwrite("fail.jpg", img)
        #    img = cv2.imread("fail.jpg")
        #    res = cenline_extract(img, "points")
        #    if res is not None:
        #        x1, y1, x2, y2 = res
        #        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
   
        #cv2.imshow("src", img)
        ##cv2.imshow("tar", res_img)
        #cv2.waitKey(3)
        
        #publish action
        tau1, tau2, tau3, tau4, tau5 = action
        a_msg = Float64MultiArray()
        a_msg.data = [tau1, tau2, tau3, tau4, tau5]
        self.Thruster_pub.publish(a_msg)
        
        #run to next frame
        rospy.sleep(0.1)

        #subscribe new state
        self.state = np.append(self.State_p.p, self.State_v.v)

        # get reward
        x, y, z, phi, theta, psi, u, v, w, p, q, r = self.state
        img = copy.deepcopy(self.IG.cv_image)
        cv2.imwrite("/home/uwsim/workspace/results/pipeline_track/record3/img%i.jpg" % self.step, img)
        cv2.imshow("src", img)
        cv2.waitKey(3)
        rew = get_reward(img, u)
        print("rew:", rew, "\t u:", u)

        done = False
        if rew is None:
            if self.last_rew is not None:
                rew = self.last_rew
                self.last_rew = None
            else:
                done = True
        else:
            self.last_rew = rew
        

        self.step += 1

        return self.state, rew, done, {}









