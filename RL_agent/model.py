
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Qnet(nn.Module):             # Q-learning network

    def __init__(self, n_states,n_frames, n_actions,hidden1,hidden2,init_w=3e-3):
        super(Qnet, self).__init__()
        self.name = 'Qnet'
        self.n_frames = n_frames
        self.n_actions = n_actions
        self.n_kernels = 16

        # architecture
        self.conv = nn.Sequential(
            nn.Conv1d(n_frames,16,kernel_size=19, stride=1),
            nn.ReLU(),
            nn.Conv1d(16,16,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv1d(16,32,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv1d(32,32,kernel_size=3,stride=1),
            nn.ReLU(),
        )

        self.linear1 = nn.Sequential(
            nn.Linear(1344, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
        )

        self.layer_norm = nn.LayerNorm(hidden2)

        self.relu = nn.ReLU()

        self.linear2 = nn.Sequential(
            nn.Linear(hidden2 + n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self,x):

        img,cmd = x
        bs = cmd.shape[0]
        feat = self.conv(img)
        feat = feat.reshape(bs,-1)
        feat = self.linear1(feat)
        feat = self.layer_norm(feat),
        feat = self.relu(feat[0])
        feat_cat = torch.cat([feat,cmd],-1)
        out = self.linear2(feat_cat)
        return out




class DwelingQnet(nn.Module):

    def __init__(self, n_states,n_frames, n_actions,hidden1,hidden2,init_w=3e-3):
        super(DwelingQnet, self).__init__()
        self.name = 'DwelingQnet'
        self.n_frames = n_frames
        self.n_actions = n_actions
        self.n_kernels = 16

        # architecture
        self.conv = nn.Sequential(
            nn.Conv1d(n_frames,self.n_kernels,kernel_size=7,stride=2),#
            nn.ReLU(),#
            nn.Conv1d(self.n_kernels,self.n_kernels,kernel_size=5,stride=2),
            nn.ReLU(),
            nn.Conv1d(self.n_kernels,1,kernel_size=3,stride=2) ,
            nn.ReLU(),
        )
        hidden = 55 
        self.blend = nn.Sequential(
            nn.Linear(hidden, hidden1),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, self.n_actions)
        )


    def forward(self,x):
        img,cmd = x
        bs = cmd.shape[0]
        feat = self.conv(img)
        feat = feat.view(bs, -1)
        feat = self.blend(torch.cat([feat,cmd],-1))
        vals = self.value_stream(feat)
        adv = self.advantage_stream(feat)
        qvals = vals + (adv - adv.mean())
        
        return qvals


