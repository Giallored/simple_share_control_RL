from shapely.geometry import LineString,Point
from shapely import Polygon
import matplotlib.pyplot as plt
from collections import namedtuple

class LabirintMaps:
    
    def __init__(self):

        self.height = 6
        self.width = 4
        self.obs_radius = 0.3
        self.Map = namedtuple('Map',field_names=[
                                                 'walls',
                                                 'obstacles',
                                                 'check_points',
                                                 'goal'])
        
        self.obs_list = [
            [-self.width*0.25,self.height*0.3],
            #[-self.width*0.4,self.height*0.3],
            [-self.width*0.1,self.height*0.7],
            [self.width*0.4,self.height*0.4],
            [self.width*0.3,self.height*0.2]
        ]
        self.goal = [self.width/4,0]
        self.check_points = [
                [-self.width/4,self.height-2.5],
                [self.width/4,self.height-2.5]
        ]


    
    def get_walls_mesh(self):
        a = [-self.width/2,-1]
        b = [self.width/2,-1]
        c = [self.width/2, -1+self.height]
        d = [-self.width/2, -1+self.height]
        ext_walls = [LineString([a,b]),LineString([a,d]),LineString([c,b]),LineString([c,d])]

        A = Point([0,-1])
        B = Point([0, self.height-3])

        wall = LineString([A,B])
        return [*ext_walls,wall]
    
    def get_obstacles_mesh(self):
        
        mesh_list = []
        for o in self.obs_list:
            x,y = o
            p = Point(x,y)
            c = p.buffer(self.obs_radius).boundary
            mesh_list.append(c)
        return mesh_list

    def sample_map(self):
        walls = self.get_walls_mesh()
        obstacles = self.get_obstacles_mesh()
        return self.Map(walls,obstacles,self.check_points,self.goal)
    
    def plot(self,map,robot):
        plt.clf()
        plt.axis('equal')
        fig = plt.gcf()
        ax = plt.gca()
        # plot the walls (lines)
        for w in map.walls:
            ax.plot(*w.xy,linewidth=1,color='blue')
        # plot obstacles (cylinders)
        for o in map.obstacles:
            ax.plot(*o.xy,linewidth=3,color='black') 
        for cp in map.check_points:
            ax.scatter(*cp,color = 'red')
        ax.scatter(*map.goal,color = 'green')
        robot, = ax.plot(*robot.exterior.xy,color='red',linewidth=1)
        fig.canvas.draw()
        fig.canvas.flush_events()

    def init_plot(self):
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.canvas.draw()
        fig.canvas.flush_events()


