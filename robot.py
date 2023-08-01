import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely import Polygon


class Robot():
    
    def __init__(self,size:tuple,x0):
        w,h = size
        self.sizes = [w,h,np.sqrt((h/2)**2+(w/2)**2)]
        self.alpha = np.pi-np.arctan2(w,h)
        self.state = np.array(x0)
        self.dstate = np.zeros(3)
        self.mesh = self.get_polygon()
        

    def get_polygon(self):
        x,y,th = self.state
        w,h,l = self.sizes
        a = th+self.alpha
        b = th-self.alpha
        shell = [
            (x+h/2*np.cos(th), y+h/2*np.sin(th)),
            (x+l*np.cos(a), y+l*np.sin(a)),
            (x+l*np.cos(b), y+l*np.sin(b))
        ]
        return Polygon(shell=shell)
    
    def move(self,v,om,dt):
        self.dstate[2] = om
        self.dstate[0] =v*np.cos(self.state[2])
        self.dstate[1] =v*np.sin(self.state[2])

        self.state = self.state + self.dstate * dt
        self.mesh = self.get_polygon()





