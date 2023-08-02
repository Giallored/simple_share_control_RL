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
from utils import get_classic_alpha,check_time

    

    
class Environment():

    def __init__(self,args,x0 = [0,0,np.pi/2]):
        
        self.agent_name=args.agent_name
        
        
        #dimentions
        self.dt = 0.05
        self.n_obs = int(args.n_obs)

        self.map_size = (10,10)
        self.robot_size = (0.2,0.3)
        #self.env_size = (1.0,1.4)
        self.obs_radius = 0.2

        #space environment
        self.map_cols = self.map_rows = 3
        self.fixed_pos = args.fixed_pos
        self.cp_list = args.check_points
       
        #print(f'There are {args.n_obs} obstacles in a grid of {len(self.grid)} ({len(ax_x)} x {len(ax_y)})')

        
        
        #initializations
        self.robot = Robot(size=self.robot_size, x0=x0)
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
        self.R_goal = 5 #-20 #-1
        self.R_col = -500 # -100
        self.R_alpha = 1 #-1
        self.R_safe= -1 #5
        self.R_cp = 10

        self.reset()    #sets the goal and obs positions at random



    def reset(self,mode='train',shuffle=True,map=None,random = True):
        self.time = 0
        
        if shuffle:
                self.obstacles,self.goal,self.check_points = self.generate_map(random)
            
        self.robot.state = [0,0,np.pi/2]
        self.last_gDist =  np.linalg.norm(np.subtract(self.goal,self.robot.state[0:2]))
        self.obs_mesh = self.get_obs_mesh(self.obstacles)
        self.scanner.reset(self.obs_mesh)
        self.ls_ranges,mask =  self.scanner.get_scan(self.robot.state)
        self.cls_point,self.cls_point_dist,self.cls_point_bear = self.scanner.ranges2clsPoint(self.ls_ranges)
        observation = self.update()
        self.is_coll = False
        self.is_goal = False
        self.is_cp =False
        self.cp_1 =True

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
    
    def get_checkpoints(self,obstacles):
        check_points = []
        for a in obstacles:
            for b in obstacles:
                dist = np.linalg.norm(np.subtract(a,b))
                if dist <= 0.9 and not a == b:
                    c = np.array([(a[0]+b[0])/2,(a[1]+b[1])/2])
                    g = np.array(self.goal)
                    o = np.array([0,0])
                    is_elegible  =(np.linalg.norm(g-c) + np.linalg.norm(c))/np.linalg.norm(g)<=1.2
                    if is_elegible:
                        check_points.append(c)
        return check_points



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
        r = self.robot.mesh
        for obs in self.obs_mesh:
            if not r.intersection(obs).is_empty:
                self.is_coll = True
            if np.sqrt((self.robot.state[0]-self.goal[0])**2+(self.robot.state[1]-self.goal[1])**2)<self.goal_toll:
                self.is_goal = True
        for cp in self.cp_list:
            if not r.intersection(Point(cp)).is_empty:
                self.is_cp = True
    
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
        elif e<0.3:
            r_safety=self.R_safe/e
        else:
            r_safety=0

        # command oriented 
        r_cmd = self.R_cmd*np.linalg.norm(np.subtract(self.cur_cmd,self.cur_cmds[0]))

        # Goal oriented
        g_dir = np.subtract(self.goal,self.robot.state[0:2])
        g_dist = np.linalg.norm(g_dir)
        delta_gDist = self.last_gDist - g_dist
       
        self.last_gDist = g_dist
        g_bear = np.arctan2(g_dir[1],g_dir[0])-self.robot.state[2]
        if self.is_goal:
            r_goal = self.R_end
        else:
            r_goal=self.R_goal*delta_gDist*(np.pi - abs(g_bear))
        
        if self.is_cp and self.cp_1:
            r_cp = self.R_cp
            self.cp_1 = False
        else:
            r_cp = 0

        r_alpha  = self.R_alpha * self.cur_alpha[0]

        self.cur_rewards = [r_safety,r_alpha,r_goal,r_cmd,r_cp]
        #print(f'rewards:\n - safe = {r_safety},\n - alpha = {r_alpha},\n - goal = {r_goal},\n - r_cp = {r_cp}')
        reward = sum(self.cur_rewards)
        return reward
    
    
    def generate_map(self,is_random=True):
        if is_random:
            obs_clear = random.uniform(0.6, 0.8)
            x_ = self.map_cols*obs_clear/2
            y_ = self.map_rows*obs_clear
            ax_x = np.linspace(-x_,x_,self.map_rows)
            ax_y = np.linspace(1.0,y_+1.0,self.map_rows)
            grid = [[x,y] for x in ax_x for y in ax_y]# if np.linalg.norm([x,y])>=1.0]
            goal_grid = [*[[x,y_+1.5] for x in ax_x]]
            obstacles = random.sample(grid, self.n_obs)
            goal = random.sample(goal_grid, 1)[0]
            check_points = self.get_checkpoints(obstacles)
        else:
            index = random.randint(0,len(self.fixed_pos))
            obstacles = self.fixed_pos[index]
            check_points = self.cp_list[index]
            goal = [0,2]

        return obstacles,goal,check_points

