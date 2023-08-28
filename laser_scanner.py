import numpy as np
from shapely.geometry import LineString,Point


class LaserScanner():
    def __init__(self):
        self.angle_min= -1.9198600053787231
        self.angle_max=  1.9198600053787231 
        self.angle_increment= 0.01 # 0.005774015095084906
        self.range_min= 0.05000000074505806
        self.range_max= 2 #25.0
        self.resolution = int((self.angle_max-self.angle_min)//self.angle_increment)
        self.angles = np.linspace(self.angle_min,self.angle_max,self.resolution)
        self.obstacles = []


    def get_scan(self,pose):
        ranges = [] # np.ones(self.resolution)*np.inf
        xy = np.array(pose[0:2])
        centre = Point(xy)
        theta = pose[-1]

        for i in range(self.resolution):
            range_i = np.inf
            #get the angle in the WF
            a_world = self.angles[i]+theta
            
            #get the line that interpolates the robot in that direction
            xy_i = xy + np.array([np.cos(a_world)*self.range_max,
                                  np.sin(a_world)*self.range_max])
            line = LineString([Point(*xy), Point(*xy_i)])
            #get the obstacles you are able to see
            for o in self.obstacles:
                points = o.intersection(line)
                if not points.is_empty:
                    try:
                        point_list = list(points.geoms)
                        for p_i in point_list:
                            dist_i = centre.distance(p_i)
                            if dist_i<=min(self.range_max,range_i): 
                                range_i = dist_i
                    except:
                        p_i = points
                        range_i = centre.distance(p_i)
            ranges.append(range_i)      
        mask = self.get_mask(ranges,3)
        return ranges,mask


    def ranges2points(self,ranges):
        pointCloud = []
        for r,a in zip(ranges,self.angles):
            if r>=self.range_min and r<=self.range_max:
                p = (r*np.cos(a),r*np.sin(a))
                pointCloud.append(p)
        return pointCloud


    def ranges2clsPoint(self,ranges):
        r = min(ranges)
        if r>self.range_max:
            return [None,None],100000,None
        i = ranges.index(r)
        a = self.angles[i]
        cls_point = [r*np.cos(a),r*np.sin(a)]
        return cls_point,r,a


    def get_mask(self,ranges,max_dist):
        mask=[]
        for r in ranges:
            if r>max_dist:
                mask.append(0)
            else:
                mask.append(1)
        return mask


    def reset(self, obs_mesh):
        self.obstacles = obs_mesh


    def preproces(self,ranges):
        ranges = np.clip(ranges,self.range_min,self.range_max)
        new_ranges = np.ones(ranges.shape)*self.range_max - ranges
        return new_ranges
