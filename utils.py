import numpy as np
from scipy.spatial.transform import Rotation
import time

import sys
import os
import matplotlib.pyplot as plt
import pickle

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot(goal,obs_list,cp_list,robot,dt):
    plt.clf()

    fig = plt.gcf()
    ax = plt.gca()
    for obs in obs_list:
        ax.plot(*obs.xy,'b-',linewidth=1)
    try:
        for cp in cp_list:
            ax.scatter(*cp,color='red')
    except:
        pass
    ax.scatter(*goal,color='green')
    plt.axis('equal')
    robot, = ax.plot(*robot.exterior.xy,color='red',linewidth=1)
    fig.canvas.draw()
    fig.canvas.flush_events()
    #time.sleep(dt)

def check_time(init_t,section):
    t = time.time()
    dt = t - init_t
    print(f'Section: {section},  time: {dt}')
    return t


def display_results(step,agent,result,reward,loss=0,duration=0,model='',train=False):
    print(f'\nRESULTS ({model}):')
    print(f' - Result: {result}')
    print(f' - Steps: {step}')
    print(f' - Reward: {reward}')
    
    if train:
        print(f' - epsilon: {round(agent.epsilon,3)}')
        print(f' - beta: {round(agent.buffer.beta,3)}')
        print(f' - Mem. Capacity: {round(agent.buffer.get_capacity()*100,1)}%')
        print(f' - Mean loss: ',round(loss,3))
        print(f' - Duration: {round(duration,2)} sec')
    print('-'*20+'\n')



def get_classic_alpha(danger_lev:int):
    primitives = {1:((1.0,0.0,0.0),'U'),
                2:((0.75,0.125,0.125),'CA'),
                3:((0.5,0.25,0.25),'CA'),
                4:((0.25,0.365,0.365),'CA'),
                5:((0.0,0.5,0.5),'CA')}
    if danger_lev == 5:
        return (0,1.0,0.0) #(0,0.5,0.5) 
    else:
        return  (1.0,0.0,0.0)
    a,tag = primitives[danger_lev]
    return a


def get_folder(name,env_name,parent_dir,mode,model2load=''):
    parent_dir = os.path.join(parent_dir,name)
    if mode =='train':
        if not model2load == "":
            return os.path.join(parent_dir,model2load)
        else:
            return  get_output_folder(parent_dir,env_name)
    elif mode =='test':
        if model2load == "":
            model2load = input('What model to load? (check "weights" folder):\n -> ')
        return os.path.join(parent_dir,model2load)
    else:
        return None

def get_output_folder(parent_dir, env_name):

    os.makedirs(parent_dir, exist_ok=True)
               
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


class Plot():
    def __init__(self,parent_dir,goal='',env='',name=''):
        self.name=name
        self.type = type
        self.dir = os.path.join(parent_dir,self.name)
        os.makedirs(self.dir, exist_ok=True)
        #initializations
        self.timesteps=[]
        self.usr_cmd=[]
        self.ca_cmd=[]
        self.ts_cmd=[]
        self.alpha=[]
        self.cmd=[]
        self.obs_poses={}
        self.ranges = {}
        self.goal=goal
        self.env=env
        
    
    def store(self,t,usr_cmd,ca_cmd,ts_cmd,alpha,cmd):
        self.timesteps.append(t)
        self.usr_cmd.append(usr_cmd)
        self.ca_cmd.append(ca_cmd)
        self.ts_cmd.append(ts_cmd)
        self.alpha.append(alpha)
        self.cmd.append(cmd)

    def close(self):
        plt.close('all')

    def save_dict(self):
        dict = {
            'type':self.type,
            'timesteps':self.timesteps,
            'usr_cmd':self.usr_cmd,
            'ca_cmd':self.ca_cmd,
            'ts_cmd':self.ts_cmd,
            'cmd':self.cmd,
            'alpha':self.alpha,
            'env':self.env,
            'goal':self.goal,
            'obs':self.obs_poses,
            'ranges':self.ranges
        }
        where = os.path.join(self.dir,'plot_dict.pkl')
        with open(where, 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('DICTS saved in: \n',where)

    def load_dict(self,dict):
        self.timesteps=dict['timesteps']
        self.usr_cmd=dict['usr_cmd']
        self.ca_cmd=dict['ca_cmd']
        self.ts_cmd=dict['ts_cmd']
        self.alpha=dict['alpha']
        self.cmd=dict['cmd']


def clamp_angle(x):
    x = (x+2*np.pi)%(2*np.pi)
    if x > np.pi+0.00001:
        return x -2*np.pi 
    else:
        return x


def get_checkpoints(obstacles,goal):
    check_points = []
    for a in obstacles:
        for b in obstacles:
            dist = np.linalg.norm(np.subtract(a,b))
            if dist <= 0.9 and not a == b:
                c = np.array([(a[0]+b[0])/2,(a[1]+b[1])/2])
                g = np.array(goal)
                o = np.array([0,0])
                is_elegible  =(np.linalg.norm(g-c) + np.linalg.norm(c))/np.linalg.norm(g)<=1.05
                if is_elegible:
                    check_points.append(c)
    return check_points


def write_console(header,alpha,a_opt,danger,lr,dt):
    l = [header,
        ' - Alpha = ' + str(alpha),
        ' - Alpha_opt = ' + str(a_opt),
        ' - Danger lev = ' + str(danger),
        ' - laerning rate = ' + str(round(lr,5)),
        ' - dt = ' + str(dt)]
    for _ in range(len(l)):
        sys.stdout.write("\x1b[1A\x1b[2K") # move up cursor and delete whole line
    for i in range(len(l)):
        sys.stdout.write(l[i] + "\n") # reprint the lines



def make_video(image_folder,vids_folder,name,dt=0.1):

    video_name = name+'.mp4'
    video_path = os.path.join(vids_folder,video_name)

    img_list = sorted(os.listdir(image_folder))
    images = [img for img in img_list if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 1/dt, (width,height))

    for image in images: 
        img = cv2.imread(os.path.join(image_folder, image))
        cv2.imshow('image', img)
        video.write(img)

    video.release()

    print('Video saved as: ',video_path)