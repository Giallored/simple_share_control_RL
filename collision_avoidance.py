import numpy as np
import matplotlib.pyplot as plt
from utils import clamp_angle

class Collision_avoider():
    def __init__(self, delta=0.7,K_lin=2.0,K_ang=5.0,k_r=50.0):
        self.th_dist=delta   #distance threshold
        self.K_lin= K_lin
        self.K_ang= K_ang
        self.k_rr = k_r
        self.k_rt= 5.0
        self.d_th = 0.2 # smallest distance
        self.gamma = 2
        self.frames={}
        self.f_i = 0
        self.max_v = 0.8    
        self.min_v = -0.8
        self.max_om = 1.0
        self.min_om = -1.0

        
    def d_Ur(self,X_p,X):
        x,y=X
        x_p,y_p=X_p

        ni=np.sqrt((x_p-x)**2 + (y_p-y)**2)
        #ni = np.linalg.norm(X_p-X)
        #dU_x = (self.k_r/self.gamma) * (x-x_p) * (1/math.sqrt((x-x_p)**2+(y-y_p)**2) - 1/self.th_dist)**(self.gamma-1)
        #dU_y = (self.k_r/self.gamma) * (y-y_p) * (1/math.sqrt((x-x_p)**2+(y-y_p)**2) - 1/self.th_dist)**(self.gamma-1)
        dU_x = -self.gamma * self.k_r * (x_p-x) * (1/ni - 1/self.th_dist)**(self.gamma-1) / ni**3
        dU_y = -self.gamma * self.k_r * (y_p-y) * (1/ni - 1/self.th_dist)**(self.gamma-1) / ni**3

        return [dU_x,dU_y]

    def get_cmd(self,X_obs,dist,theta):
        X_obs = np.array(X_obs)
        if X_obs[0] ==None:
            return [0.0,0.0],[0.0,0.0]
        theta =clamp_angle(theta) 
        sign = - np.sign(theta)
        if sign==0.0: sign = 1
        R = np.array([[0,1],[-1,0]]) *sign
        dU = self.dU_rt(dist)
        F_r = dU*X_obs
        #rotational component
        #dU_r=self.d_Ur(X_obs,[0,0])   
        #F_r = np.array([dU_r[1],-dU_r[0]]) 
        #dtheta_d = np.arctan2(*F_r)
        #print('before: ',dtheta_d/np.pi*180 )
        #dtheta_d = clamp_angle(dtheta_d)
        #print('after: ',dtheta_d/np.pi*180)
        #om_cmd = np.clip(-self.K_ang*(dtheta_d),self.min_om,self.max_om)
        #print('om_cmd: ',om_cmd)
        #input()
        #v_cmd = np.clip(self.K_lin*(theta/np.pi)**2,self.min_v,self.max_v)
        #cmd_r = [v_cmd,om_cmd]

        #Rotational component
        dtheta = np.arctan2(*F_r)
        dtheta = clamp_angle(dtheta)*sign

        v_cmd = self.K_lin*(theta/np.pi)**2
        v_cmd =  np.clip(v_cmd,self.min_v,self.max_v)
        om_cmd = np.clip(-self.K_ang*(dtheta) ,self.min_om,self.max_om)
        cmd_r = [v_cmd,om_cmd]
        


        #translational component
        #F_t = dU*X_obs
        F_t = F_r @ R 
        dtheta_d = np.arctan2(*F_t)
        dtheta_d = clamp_angle(dtheta_d)
        om_cmd = np.clip(self.K_ang*(dtheta_d),self.min_om,self.max_om)
        v_cmd = np.clip(self.K_lin*(theta/np.pi)**2,self.min_v,self.max_v)
        cmd_t = [v_cmd,om_cmd]


        return cmd_r, cmd_t
    
    def dU_rt(self,dist):
        return self.k_rt * (1/dist - 1/self.d_th) * 1/(dist**3) 

    
    def get_cls_point(self,point_cloud):
        if not point_cloud:
            return [None,None]
        min_dist=1000
        cls_point=point_cloud[0]
        for p in point_cloud:
            dist = np.linalg.norm(p)
            if dist<min_dist:
                min_dist=dist
                cls_point = p
        return np.array(cls_point)