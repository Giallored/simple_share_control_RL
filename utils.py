import rospy
from geometry_msgs.msg import Twist
import numpy as np
from scipy.spatial.transform import Rotation
import time
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import SetModelState

import sys
import os
import matplotlib.pyplot as plt
import pickle

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
        print(f' - Epsilon: {round(agent.epsilon,3)}')
        print(f' - Mem. Capacity: {round(agent.buffer.get_capacity()*100,1)}%')
        print(f' - Mean loss: ',round(loss,3))
        print(f' - Duration: {round(duration,2)} sec')
    print('-'*20+'\n')



def get_classic_alpha(danger_lev:int):
    primitives = {1:((1.0,0.0,0.0),'U'),
                2:((0.75,0.25,0.0),'CA'),
                3:((0.5,0.5,0.0),'CA'),
                4:((0.25,0.75,0.0),'CA'),
                5:((0.0,1.0,0.0),'CA')}
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


class TrainData():
    def __init__(self,parent_dir,env='',name=''):
        self.name=name
        self.dir = parent_dir   
        self.epochs = []
        self.rewards = []
        self.loss = []
        self.lr = 0.0001
        self.epsilon=0.7
        self.env=env

    def store(self,epoch,reward,loss,lr=0.001,eps=0.7):
        self.epochs.append(epoch)
        self.rewards.append(reward)
        self.loss.append(loss)
        self.lr = lr
        self.epsilon=eps

    def save_dict(self):
        dict = {
            'epoch':self.epochs,
            'reward':self.rewards,
            'loss':self.loss,
            'lr':self.lr,
            'eps':self.epsilon
        }
        where = os.path.join(self.dir,'train_dict.pkl')
        with open(where, 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('DICTS saved in: \n',where)

    def plot_and_save(self):
        fig1 = plt.figure()
        plt.plot(self.epochs,self.rewards,'r-')
        path = os.path.join(self.dir,'rewards.png')
        plt.savefig(path)

        fig2 = plt.figure()
        plt.plot(self.epochs,self.loss,'r-')
        path = os.path.join(self.dir,'loss.png')
        plt.savefig(path)

        print('PLOTS saved in: \n',self.dir)
        self.close()

    def load_dict(self,dict):
        self.epochs = dict['epoch']

        self.rewards = dict['reward']
        self.loss = dict['loss']
        self.lr =dict['lr']
        self.eps =dict['eps']

    def close(self):
        plt.close('all')
    
        

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

    if x > np.pi:
        return x -2*np.pi 
    else:
        return x

    #while x>np.pi:
    #    x-=2*np.pi
    #while x<-np.pi:
    #    x+=2*np.pi
    #return x


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


class Contact():
    def __init__(self,msg:ContactsState):
        self.state=msg.states
    
    def check_contact(self):
        if self.state==[]:
            return None,None
        else:
            obj_1 = self.clean_name(self.state[0].collision1_name)
            obj_2 = self.clean_name(self.state[0].collision2_name)
            return obj_1,obj_2

    def clean_name(self,name):
        final_name=''
        for l in name:
            if l == ':':
                break
            else:
                final_name+=l
        return final_name 


#for processing pointclouds to images
def pc2img(pc,defi=2,height=300,width=400):
    pc = np.around(pc,decimals=defi)*10**defi#clean
    pc[:,1]+=width/2      #translate
    #print((pc[:,0]>=0) & (pc[:,0]<=300)&(pc[:,1]>=0) & (pc[:,1]<=height))
    pc = pc[(pc[:,0]>0)  & (pc[:,0]<height)  & (pc[:,1]>0)  & (pc[:,1]<width)] #crop
    pc=np.array(pc).astype('int') 
    rows = pc[:,0]
    cols = pc[:,1]
    img = np.zeros((height,width))
    img[rows,cols]=255
    kernel = np.ones((5,5),np.uint8)#
    img = cv.dilate(img,kernel,iterations = 1)
    #img = cv.resize(img, (400,300), interpolation = cv.INTER_AREA)
    return img






