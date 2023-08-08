from collections import namedtuple
from utils import *
import random

class Maps:
    def __init__(self,n_obs,grid_size = (3,3)):
        self.Map = namedtuple('Map',field_names=['obstacles','check_points','goal'])
        self.n_obs =n_obs
        #random maps params
        self.x_min = -0.4
        self.x_max = 0.4
        self.y_min = 1.0
        self.y_max = 2.5
        self.min_clear = 0.7

        #evaluation maps
        self.eval_maps = self.get_eval_maps()


        #grid maps
        obs_clear = 0.7
        rows,cols = grid_size
        x_ = cols*obs_clear/2
        y_ = rows*obs_clear
        ax_x = np.linspace(-x_,x_,cols)
        ax_y = np.linspace(1.0,y_+1.0,rows)
        self.grid = [[x,y] for x in ax_x for y in ax_y]# if np.linalg.norm([x,y])>=1.0]
        self.goal_grid = [*[[x,y_+1.5] for x in ax_x]]

    def get_sigle_map(self):
        obs = [
        [0.15,1.75],[-0.35,1.0],[0.5,1.0],[-0.5,2.25]
        ]
        goal =[0.5,2.5]
        cp = [[0.075, 1.   ],[-0.175,  2.   ]]
        return self.Map(obs,cp,goal)

    def get_eval_maps(self):
        eval_maps = {}
        eval_obs = [
        [[0.35,1.0],[-0.35,1.5],[0.35,2]],
        [[-0.35,1.0],[0.35,1.5],[-0.35,2]],
        [[self.x_max,0.5],[-0.2,1.0],[self.x_max,1.5]],
        [[self.x_min,0.5],[0.2,1.0],[self.x_min,1.5]],
        [[-0.5,0.5],[0.5,0.5],[0.0,1.25]],
        [[-0.5,0.5],[0.5,0.5],[0.0,1.25]],
        [[-0.5,1.5],[0.5,1.5],[0.0,0.75]],
        ]
        eval_goals = [
                    [0,2.5],
                    [0,2.5],
                    [self.x_max,2.0],
                    [self.x_min,2.0],
                    [self.x_min,2.0],
                    [self.x_max,2.0],
                    [0.0,2.0],
                ]
        for i in range(len(eval_obs)):
            o = eval_obs[i]
            g = eval_goals[i] 
            c =get_checkpoints(o,g)
            map = self.Map(o,c,g)
            eval_maps[i] = map
        return eval_maps
    
    def get_random_map(self):
        obs_1 = [1.0,random.uniform(self.y_min,self.y_max)]
        obstacles = [obs_1]
        for i in range(self.n_obs-1):
            clear = False
            while not clear:
                obs_i = np.array([random.uniform(self.x_min, self.x_max),random.uniform(self.y_min,self.y_max)])
                clear = all(np.linalg.norm(obs_i-np.array(o)) > self.min_clear for o in obstacles)
            obstacles.append(obs_i.tolist())
        goal = [0,self.y_max+0.5]
        check_points = []# get_checkpoints(obstacles,goal)
        return self.Map(obstacles,check_points,goal)
    

    def get_grid_map(self):
        obstacles = random.sample(self.grid, self.n_obs)
        goal = random.sample(self.goal_grid, 1)[0]
        check_points = get_checkpoints(obstacles,goal)
        return self.Map(obstacles,check_points,goal)
    
    def get_labirint_map(self):
        pass

    
    def get_map(self, type='random'):
        if type=='grid':
            return self.get_grid_map()
        elif type=='eval':
            return random.choice(self.eval_maps)
        elif type =='single':
            return self.get_sigle_map()
        else:
            return self.get_random_map()
            





            