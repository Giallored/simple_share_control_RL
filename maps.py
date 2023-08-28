from collections import namedtuple
from utils import *
import random
from shapely.geometry import LineString,Point

class Maps:
    def __init__(self,n_obs,grid_size = (3,3)):
        self.Map = namedtuple('Map',field_names=[
                                                 'walls',
                                                 'obstacles',
                                                 'check_points',
                                                 'start',
                                                 'goal'])
        
        #self.Map = namedtuple('Map',field_names=['obstacles','check_points','goal'])
        self.n_obs =n_obs

        #random maps params
        self.x_min = -0.75
        self.x_max = 0.75
        self.y_min = 0.0
        self.y_max = 4.0
        self.min_clear = 0.8

        #grid maps
        self.grid_size = grid_size
        obs_clear = 0.7
        rows,cols = grid_size
        x_ = cols*obs_clear/2
        y_ = rows*obs_clear
        ax_x = np.linspace(-x_,x_,cols)
        ax_y = np.linspace(1.0,y_+1.0,rows)
        self.grid = [[x,y] for x in ax_x for y in ax_y]# if np.linalg.norm([x,y])>=1.0]
        self.goal_grid = [*[[x,y_+1.5] for x in ax_x]]

        #labirinth maps
        self.height = 8
        self.width = 6
        self.obs_radius = 0.2


    # obstacles spawn randomly in a minimum pre-defined distance between each other
    def get_random_map(self):
        a = [self.x_min-1,self.y_min-1]
        b = [self.x_max+1,self.y_min-1]
        c = [self.x_max+1, self.y_max +1.5]
        d = [self.x_min-1, self.y_max +1.5]
        ext_walls = [LineString([a,d]),LineString([c,d]),LineString([c,b])]
        obs_1 = [1.0,random.uniform(self.y_min,self.y_max)]
        obs_xy = [obs_1]
        for i in range(self.n_obs-1):
            clear = False
            while not clear:
                obs_i = np.array([random.uniform(self.x_min, self.x_max),random.uniform(self.y_min,self.y_max)])
                clear = all(np.linalg.norm(obs_i-np.array(o)) > self.min_clear for o in obs_xy)
            obs_xy.append(obs_i.tolist())
        obstacles = self.point2Obs(obs_xy)
        start = [random.uniform(self.x_min,self.x_max),self.y_min-1.0,np.pi/2]
        goal = [random.uniform(self.x_min,self.x_max),self.y_max+0.5]
        check_points = [] # get_checkpoints(obs_xy,goal)
        return self.Map(ext_walls,obstacles,check_points,start,goal)
    
    # obstacles spawn randomly in a grid
    def get_grid_map(self):
        obstacles = random.sample(self.grid, self.n_obs)
        obstacles = self.point2Obs(obstacles)
        start = [0,0,np.pi/2]
        goal = random.sample(self.goal_grid, 1)[0]
        check_points = get_checkpoints(obstacles,goal)
        return self.Map([],obstacles,check_points,start,goal)

     # labirinth map
    def get_lab_map(self):
        a = [-self.width/2,-1]
        b = [self.width/2,-1]
        c = [self.width/2, -1+self.height]
        d = [-self.width/2, -1+self.height]
        ext_walls = [LineString([a,b]),LineString([a,d]),LineString([c,b]),LineString([c,d])]
        A = Point([0,-1])
        B = Point([0, self.height-3])
        wall = LineString([A,B])
        walls = [*ext_walls,wall]
        obstacles = self.point2Obs(self.obs_list)
        start = [-self.width/4,0,np.pi/2]
        goal = [self.width/4,0]
        check_points = [
                [-self.width/4,self.height-2.5],
                [self.width/4,self.height-2.5]
        ]
        return self.Map(walls,obstacles,check_points,start,goal)

    def init_plot(self):
        plt.ion()
        #fig, ax = plt.subplots(figsize=(10, 10))
        self.fig, self.axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def plot(self,map,robot,alpha):
        fig = self.fig 
        ax1,ax2 = self.axs
        ax1.clear()
        ax2.clear()
        ax1.set_aspect('equal')

        # plot the walls (lines)
        for w in map.walls:
            ax1.plot(*w.xy,linewidth=1,color='blue')
        # plot obstacles (cylinders)
        for o in map.obstacles:
            ax1.plot(*o.xy,linewidth=0.5,color='black') 
        for cp in map.check_points:
            ax1.scatter(*cp,color = 'red')
        ax1.scatter(*map.goal,color = 'green')
        #plot the robot
        robot, = ax1.plot(*robot.exterior.xy,color='red',linewidth=1)

        #bar graph
        ax2.set_ylim([0.1,1.1])
        bars = ax2.bar(['a U','a CA(r)','a CA(t)'],alpha)
        bars[0].set_color('green')
        bars[1].set_color('blue')
        bars[2].set_color('red')
        fig.canvas.draw()
        fig.canvas.flush_events()


    def point2Obs(self,points):
        obstacles = []
        for o in points:
            x,y = o
            p = Point(x,y)
            c = p.buffer(self.obs_radius).boundary
            obstacles.append(c)
        return obstacles


    def sample_map(self, type='random'):
        if type=='grid':
            return self.get_grid_map()
        elif type =='labirinth':
            return self.get_lab_map()
        else:
            return self.get_random_map()
            





            