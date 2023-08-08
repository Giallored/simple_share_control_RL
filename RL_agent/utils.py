
import os
import torch
from torch.autograd import Variable
import numpy as np

#USE_CUDA = False#torch.cuda.is_available()
#FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def power_decay_schedule(n: int,
                    eps_decay: float,
                    eps_decay_min: float) -> float:
    return max(eps_decay**n, eps_decay_min)

def exponential_annealing_schedule(n, rate):
    return 1 - np.exp(-rate * n)


def to_numpy(var,use_cuda = False):
    return var.cpu().data.numpy() if use_cuda else var.data.numpy()

def to_tensor(array, use_cuda = False,volatile=False, requires_grad=False):
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    out = Variable(torch.from_numpy(array), volatile=volatile, requires_grad=requires_grad).type(dtype)
    return out
    
def to_longTensor(array, use_cuda = False,volatile=False, requires_grad=False):
    dtype = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    out = Variable(torch.from_numpy(array), volatile=volatile, requires_grad=requires_grad).type(dtype)
    return out

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class HyperParams:
    def __init__(self):
        self.hidden1=128            #hidden
        self.hidden2=128            #hidden
        self.rate=0.001
        self.warmup= 300            #time
        self.discount=0.99
        self.bsize=256 #
        self.rmsize=5000 #6000000
        self.tau= 0.001              #for the soft update
        self.validate_episodes= 20
        self.max_episode_length= 100
        self.validate_steps= 1000
        self.output_weight= 'weights'
        self.output_plot= 'plots'
        self.parent_dir= '/home/adriano/Desktop/RL_simulated_unicycle/'
        self.init_w= 0.003
        self.max_train_iter= 200000 
        self.max_epochs= 500
        self.n_frames= 3



