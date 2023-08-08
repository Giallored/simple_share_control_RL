
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#from ipdb import set_trace as debug

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)



'''
net1 = tf.compat.v1.layers.conv1d(s2, filters=32, kernel_size=19, strides=1, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net1')
net2 = tf.compat.v1.layers.conv1d(net1, filters=32, kernel_size=8, strides=4, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net2')
net3 = tf.compat.v1.layers.conv1d(net2, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net3')
net4 = tf.compat.v1.layers.conv1d(net3, filters=64, kernel_size=3, strides=1, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net4')
net4_flat=tf.reshape(net4,[-1,19*64])
net5=tf.layers.dense(net4_flat, 608, activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net5')
net6 = tf.layers.dense(net5, 128, trainable=trainable, name='net6')
net6 = tf.contrib.layers.layer_norm(net6, center=True, scale=True)
net6 = tf.nn.relu(net6)
net7_input = tf.concat([s1, a, net6], 1)
net7 = tf.layers.dense(net7_input, 64, trainable=trainable,name='net7')
net7 = tf.contrib.layers.layer_norm(net7, center=True, scale=True)
net7 = tf.nn.relu(net7)
net7 = tf.layers.dense(net7, 16, trainable=trainable,name='net7_')
net7 = tf.contrib.layers.layer_norm(net7, center=True, scale=True)
net7 = tf.nn.relu(net7)
return tf.layers.dense(net7, 1, trainable=trainable,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,maxval=3e-3),name='q_val')  # Q(s,a)

'''




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
            #nn.MaxPool1d(3,stride=1),
            #nn.BatchNorm1d(self.n_kernels),
            nn.ReLU(),
            nn.Conv1d(self.n_kernels,1,kernel_size=3,stride=2) ,
            #nn.BatchNorm1d(1),
            #nn.MaxPool1d(3,stride=1),
            nn.ReLU(),
        )
        hidden = 55 #100 #n_states + 161
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


class SparseQnet(nn.Module):

    def __init__(self, n_states, n_actions,hidden1,hidden2):
        super(SparseQnet, self).__init__()
        self.name = 'SparseQnet'
        #self.SparseLayer1 = SparseConv(1, 16, 11)
        #self.SparseLayer2 = SparseConv(16, 16, 7)
        #self.SparseLayer3 = SparseConv(16, 16, 5)
        #self.SparseLayer4 = SparseConv(16, 16, 3)
        #self.SparseLayer5 = SparseConv(16, 16, 3)
        #self.SparseLayer6 = SparseConv(16, 1, 1)

        self.SparseLayer1 = SparseConv(1, 16, 7)
        self.SparseLayer2 = SparseConv(16, 16, 5)
        self.SparseLayer3 = SparseConv(16, 16, 3)
        self.SparseLayer4 = SparseConv(16, 1, 1)
    
        self.conv1 = nn.Conv2d(1,3,3,stride = 2)
        self.conv2 = nn.Conv1d(3,1,3,stride = 2)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(165, hidden1)
        self.fc2 = nn.Linear(n_states+hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)
        print('feat: ',feat.shape)
        self.SparseLayer4(feat, mask)
        #feat, mask = self.SparseLayer5(feat, mask)
        #feat, mask = self.SparseLayer6(feat, mask)

        feat, mask = self.SparseLayer1(img, mask)
        feat, mask = self.SparseLayer2(feat, mask)
        feat, mask = self.SparseLayer3(feat, mask)
        feat, mask = self.SparseLayer4(feat, mask)

        #print('BOB: ',feat.shape)
        feat = self.relu(self.conv1(feat))
        feat = feat.squeeze(2)
        #print('0: ',feat.shape)

        feat = self.relu(self.conv2(feat))

        #print('1: ',feat.shape)
        feat = feat.reshape(bs,feat.shape[-1]*feat.shape[-2])

        #print('2: ',feat.shape)
        feat = self.relu(self.fc1(feat))
        #print('3: ',feat.shape)
        feat = self.relu(self.fc2(torch.cat([feat,cmd],-1)))
        #print('4: ',feat.shape)
        out = self.fc3(feat)
        return out



class SparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super().__init__()
        padding = kernel_size//2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        self.bias = nn.Parameter(
            torch.zeros(out_channels), 
            requires_grad=True)

        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        kernel = torch.FloatTensor(torch.ones(kernel_size,kernel_size)).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel, 
            requires_grad=False)

        self.relu = nn.ReLU(inplace=True)


        self.max_pool = nn.MaxPool2d(
            kernel_size, 
            stride=1, 
            padding=padding)

        
    def forward(self, x, mask):
        x = x*mask
        x = self.conv(x)
        normalizer = 1/(self.sparsity(mask)+1e-8)
        x = x * normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = self.relu(x)
        
        mask = self.max_pool(mask)

        return x, mask



