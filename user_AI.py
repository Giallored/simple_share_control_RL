import random
import numpy as np
from utils import *



class User():
    
    def __init__(self):
            
        #primitives
        self.primitive_v = 0.8
        self.primitive_om = 1.0
        self.stop_cmd = [0.0,0.0]

        #threshold on bearing to turn 
        self.theta_th = np.pi/4

        #paramss
        self.max_v = 0.8    
        self.min_v = -0.8
        self.max_om = 1.0
        self.min_om = -1.0
        self.k_l = 0.5
        self.k_a = 10        
        self.max_noise = 0.01 


    def get_cmd(self,pose,goal):
        dist,theta = self.get_goal_dist(pose,goal)
        cmd = self.continue_act(dist,theta)
        return cmd

    def get_goal_dist(self,pose,goal):
        dist = np.subtract(goal,pose[0:2]) #2d dist
        l_dist = np.linalg.norm(dist)
        a_dist = clamp_angle(np.arctan2(dist[1],dist[0]))
        r_theta =clamp_angle(pose[-1])
        theta = clamp_angle(a_dist-r_theta)

        return l_dist,theta
        
    def continue_act(self,dist,theta):
        noise = np.random.uniform(-self.max_noise,self.max_noise,2)
        if abs(theta)<=self.theta_th:
            v = np.clip(self.k_l*dist+noise[0],self.min_v,self.max_v)
        else:
            v = 0.0
        om = np.clip(self.k_a*theta+noise[1],self.min_om,self.max_om)
        cmd = [v,om]
        return cmd


if __name__ == '__main__':
    try:
        rospy.init_node('User', anonymous=True)
        node =User(discrete = False)
        node.main()
    except rospy.ROSInterruptException:
        pass