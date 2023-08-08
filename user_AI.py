import random
import numpy as np
from utils import *



class User():
    
    def __init__(self):
            
        #primitives
        self.straight = 1.0
        self.turn_L = 1.0
        self.turn_R = -1.0
        self.primitives = [[ self.straight,self.turn_L],[ self.straight,self.turn_R],[ self.straight,0.0],[0.0,self.turn_L],[0.0,self.turn_R]]
        self.stop_cmd = [0.0,1.0,-1.0]
        self.p_th = 0.1

        #threshold on bearing to turn 
        self.theta_th = np.pi/9

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
        #cmd = self.continue_act(dist,theta)
        cmd = self.discrete_act(dist,theta)
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
    
    def discrete_act(self,dist,theta):
        p = random.uniform(0, 1)
        if p<self.p_th:
            return random.choice(self.primitives)
        else:
            if dist >0 and abs(theta)<self.theta_th:
                v = self.straight
            else:
                v=0
            
            if theta>self.theta_th:
                om = self.turn_L
            elif theta<-self.theta_th:
                om = self.turn_R
            else:
                om=0.0
            return [v,om]

            





if __name__ == '__main__':
    try:
        rospy.init_node('User', anonymous=True)
        node =User(discrete = False)
        node.main()
    except rospy.ROSInterruptException:
        pass