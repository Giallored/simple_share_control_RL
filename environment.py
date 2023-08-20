import numpy as np
from roboticstoolbox import Unicycle
import random
from robot import Robot
from shapely.geometry import LineString,Point
from collision_avoidance import Collision_avoider
from trajectory_smoother import Trajectory_smooter
from user_AI import User
from laser_scanner import LaserScanner
import matplotlib.pyplot as plt
import time
from utils import *
from maps import Maps
from labirint_maps import LabirintMaps
from copy import deepcopy

    
class Environment():

    def __init__(self,args,x0 = [-1.5,0,np.pi/2]):
        
        self.agent_name=args.agent_name
        
        self.seed = None
        #dimentions
        self.dt = 0.2
        #self.n_obs = int(args.n_obs)

        self.map_size = (10,10)
        self.robot_size = (0.2,0.3)
        #self.env_size = (self.n_obs%3,1.4)
        self.obs_radius = 0.2

        #space environment
        self.cp_list = []
        #self.maps = Maps(self.n_obs,(3,3))
        self.maps = LabirintMaps()
        #print(f'There are {args.n_obs} obstacles in a grid of {len(self.grid)} ({len(ax_x)} x {len(ax_y)})')

        
        
        #initializations
        self.robot = Robot(size=self.robot_size, x0=x0)
        self.init_state = x0
        self.scanner = LaserScanner()
        self.time = 0
        self.cur_cmd = [0.0,0.0]
        self.usr_cur_cmd = [0.0,0.0]
        self.goal_toll= 0.3

        #controllers
        self.usr_controller = User()
        self.ca_controller = Collision_avoider()
        self.ts_controller = Trajectory_smooter(dt=self.dt)

        #rewards
        self.R_cmd = 0 #-1
        self.R_end = 1000
        self.R_goal = 5 #-1
        self.R_col = -500 # -100
        self.R_alpha = 1 #-1
        self.R_safe= -1
        self.R_cp = 100

        self.reset()    #sets the goal and obs positions at random



    def reset(self,shuffle=True,map=None,random = True):
        self.time = 0
        
        #if shuffle:
        #    self.obstacles,self.check_points,self.end_goal = self.maps.get_map(map)
        #    self.check_points.append(self.end_goal )
        #    self.goal = self.check_points.pop(0)
        if shuffle:
            self.map = self.maps.sample_map()
            walls = self.map.walls
            obs = self.map.obstacles
            self.obs_mesh = [*walls,*obs]
            self.check_points = deepcopy(self.map.check_points)
            self.end_goal = self.map.goal

            self.check_points.append(self.end_goal )
            self.goal = self.check_points.pop(0)

        
        self.robot.state = self.init_state
        self.last_gDist =  np.linalg.norm(np.subtract(self.goal,self.robot.state[0:2]))
        #self.obs_mesh = self.get_obs_mesh(self.obstacles)
        self.scanner.reset(self.obs_mesh)
        self.ls_ranges,mask =  self.scanner.get_scan(self.robot.state)
        self.cls_point,self.cls_point_dist,self.cls_point_bear = self.scanner.ranges2clsPoint(self.ls_ranges)
        observation = self.update()
        self.is_coll = False
        self.is_goal = False
        self.is_cp =False

        self.cur_cmd = [0.0,0.0]
        self.cur_cmds = [(0.0,0.0),(0.0,0.0),(0.0,0.0)]
        return observation
    
        
    def step(self,cmd):
        v,om=cmd
        self.cur_cmd = cmd
        self.time += self.dt
        self.ts_controller.store_action(self.time,cmd)
        self.robot.move(v,om,self.dt)
        observation = self.update()
        reward = self.get_reward()
        done = self.is_coll or self.is_goal
        return observation,reward,done


    def change_seed(self,seed):
        random.seed(seed)
        self.usr_controller.change_seed(seed)



    def update(self):
        self.ls_ranges,mask =  self.scanner.get_scan(self.robot.state)
        self.cls_point,self.cls_point_dist,self.cls_point_bear = self.scanner.ranges2clsPoint(self.ls_ranges)
        if self.agent_name == 'SparseQnet':
            observation = np.hstack([self.ls_ranges,mask])
        else:
            observation = self.scanner.preproces(self.ls_ranges)
        self.check()
        self.get_danger()
        return observation
        
    
    def get_cmds(self):
        #point_cloud = self.scanner.ranges2points(self.ls_ranges)
        usr_cmd = self.usr_controller.get_cmd(self.robot.state,self.goal)
        ca_cmd1,ca_cmd2 = self.ca_controller.get_cmd(self.cls_point,self.cls_point_dist,self.cls_point_bear)
        #cls_obs,cls_dist,cls_bear = self.get_RW_cls_point()
        #ca_cmd1,ca_cmd2 = self.ca_controller.get_cmd(cls_obs,cls_dist,cls_bear)
        #ts_cmd = self.ts_controller.get_cmd(self.time)
        self.cur_cmds = (usr_cmd,ca_cmd1,ca_cmd2)
        return self.cur_cmds
    




    def get_danger(self):
        dist = self.cls_point_dist
        theta =  self.cls_point_bear
        theta_th = np.arcsin(0.5/np.clip(dist,0.5,np.inf))
        if dist*10>5:
            self.danger = 1
        else:
            self.danger =  int((0.5-dist+0.1)//0.1+1)

        if dist<1.0 and dist>=0.5:
            if abs(theta)<theta_th:
                self.danger =3
            else:
                self.danger =2
        elif dist<0.5:
            if abs(theta)<theta_th:   
                self.danger= 5
            else:
                self.danger= 4
        else:
            self.danger = 1

    def update_alpha(self,current,target):
        self.cur_alpha = current
        self.target_alpha = target

    def get_state(self,observation,prev_alpha):
        state_vars = np.hstack([self.cur_cmds[0],
                                self.cur_cmds[1],
                                prev_alpha,
                                self.cur_cmd])
        return [observation,state_vars]

    def check(self):
        self.is_coll = False
        self.is_goal = False
        self.is_cp = False
        r = self.robot.mesh
        for obs in self.obs_mesh:
            if not r.intersection(obs).is_empty:
                self.is_coll = True
            if np.sqrt((self.robot.state[0]-self.goal[0])**2+(self.robot.state[1]-self.goal[1])**2)<self.goal_toll:
                if self.check_points:
                    self.is_cp = True
                    self.goal = self.check_points.pop(0)
                else:
                    self.is_goal =True
                


    
    def plot_xy(self):
        robot, = self.ax.plot(*self.robot.mesh.exterior.xy,linewidth=1)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def get_obs_mesh(self,obs_list):
        mesh_list = []
        for o in obs_list:
            x,y = o
            p = Point(x,y)
            c = p.buffer(self.obs_radius).boundary
            mesh_list.append(c)
        return mesh_list


    def get_reward(self):

        # safety oriented
        e = self.cls_point_dist
        if self.is_coll:
            r_safety = self.R_col
        elif e<0.5:
            r_safety=self.R_safe/e
        else:
            r_safety=0

        # Goal oriented
        g_dir = np.subtract(self.goal,self.robot.state[0:2])
        g_dist = np.linalg.norm(g_dir)
        delta_gDist = self.last_gDist - g_dist
       
        self.last_gDist = g_dist
        g_bear = np.arctan2(g_dir[1],g_dir[0])-self.robot.state[2]
        if self.is_goal:
            r_goal = self.R_end
        elif self.is_cp:
            r_goal = self.R_cp
        else:
            r_goal=self.R_goal*delta_gDist*(np.pi - abs(g_bear))

        r_alpha  =  self.R_alpha * self.cur_alpha[0]

        self.cur_rewards = [r_safety,r_alpha,r_goal]
        #print(f'rewards:\n - safe = {r_safety},\n - alpha = {r_alpha},\n - goal = {r_goal}')#,\n - r_cp = {r_cp}')
        reward = sum(self.cur_rewards)
        return reward

    def collision_forecast(self,v,om):
        if self.danger ==5:
            _,_,r_mesh = self.robot.move_simulate(v,om,self.dt)
            for obs in self.obs_mesh:
                if not r_mesh.intersection(obs).is_empty:
                    return True
        else:
            return False

