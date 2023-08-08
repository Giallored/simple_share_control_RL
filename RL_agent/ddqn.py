
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from collections import deque
from RL_agent.model import DwelingQnet,SparseQnet,Qnet
from RL_agent.ERB import Prioritized_ERB
from RL_agent.utils import *
from copy import deepcopy
from torch.nn import Softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau,LambdaLR,StepLR
from statistics import mean
from RL_agent.ERB_new import PrioritizedReplayBuffer
# from ipdb import set_trace as debug


class DDQN(object):
    def __init__(self, n_states, n_frames, action_space,args,is_training=True):
        self.name = 'ddqn'
        self.n_states = n_states
        self.n_actions= len(action_space)
        self.action_space = action_space
        print(f'There are {self.n_actions} primitive actions.')

        self.scheduler_type = 'StepLR'
        self.n_frames = n_frames
        self.lr = args.rate
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2
                            }
        #self.network = SparseQnet(self.n_states, self.n_actions,**net_cfg)
        self.network = Qnet(self.n_states,self.n_frames, self.n_actions,**net_cfg)
        #self.network = DwelingQnet(self.n_states,self.n_frames, self.n_actions,**net_cfg)
        self.target_network = deepcopy(self.network)

        _optimizer_kwargs = {
            "lr": self.lr,
            "betas":(0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False,
        }
        self.optimizer  = Adam(self.network.parameters(), **_optimizer_kwargs)
        self.loss= nn.MSELoss()

        if  self.scheduler_type == 'StepLR':
            self.lr_scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)
        elif self.scheduler_type == 'ReduceLROnPlateau':
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=3, threshold=5,
                               threshold_mode='rel', cooldown=0, min_lr=0.00001, eps=1e-08)
            self.lr_scheduler.step(-1000)
        elif self.scheduler_type == 'LambdaLR':
            self.window_len = args.lambda_window
            self.last_lambda_mean = 100000
            self.lr_coeff=1.0
            self.loss_window = deque([self.last_lambda_mean]*self.window_len,maxlen=self.window_len)
            self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=self.lambda_rule)
            

        hard_update(self.target_network, self.network) # Make sure target is with the same weight
        
        #Create replay buffer
        #self.buffer = Prioritized_ERB(self.n_frames,memory_size = args.rmsize)
        self.buffer = PrioritizedReplayBuffer(capacity = args.rmsize,
                                              o_shape = (self.n_frames,383),
                                              s_shape = (self.n_states,), 
                                              a_shape=(1,),
                                              alpha=0.5) 
        # Hyper-parameterssigma
        self.batch_size = args.bsize
        self.gamma = args.discount
        self.tau = args.tau
        self.epsilon_decay=args.epsilon_decay
        self.epsilon = args.epsilon
        self.epsilon_min=0.01
        self.is_training = is_training
        self.policy_freq=2 #delayed actor update

        self.epsilon_decay_schedule =lambda n: power_decay_schedule(n, 
                                                    eps_decay = self.epsilon_decay,
                                                    eps_decay_min=self.epsilon_min)

        print(f'Hyper parmaeters are:\n - epsilon = {self.epsilon}\n - epsilon decay = {self.epsilon_decay}\n - learning rate = {self.lr}')
        
        self.train_iter = 0
        self.sync_frequency=200
        self.update_frequency = 4
        self.max_iter = args.max_train_iter

        #initializations
        self.a_t = (1.0,0.0,0.0) # Most recent action
        self.episode_loss=0.
        

        self.use_cuda = torch.cuda.is_available()
        #self.use_cuda = False
        if self.use_cuda: self.cuda()

    def set_hp(self,hp):
        self.epsilon = hp.eps
        self.optimizer  = Adam(self.network.parameters())
        self.last_lambda_mean = hp.loss[-1]
        #self.lr_scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.9)
        print('Epsilon set to ',hp.eps)



    def reset(self,init_state,init_act):
        init_obs,init_cmd = init_state

        self.observations=deque([init_obs]*self.n_frames,maxlen=self.n_frames)
        self.sVars = init_cmd
        self.a_t=np.array(init_act)
        self.episode_loss=0.

    def e_greedy(self):
        p = np.random.random()
        if p < self.epsilon:
            return 'explore'
        else:
            return 'exploit'
    
    def get_lr(self):
        if self.scheduler_type=='ReduceLROnPlateau':
            return self.lr_scheduler._last_lr[0]
        else:
            return self.lr_scheduler.get_last_lr()[0]

    def random_action(self):
        action = random.sample(self.action_space,1)[0] #np.random.choice(self.action_space)
        self.a_t = action
        return action
    


    def select_action(self, s_t):
        obs_t,svar_t=s_t
        
        #assemble the state
        observation=deepcopy(self.observations)
        observation.append(obs_t)
        observation = np.array([observation])
        bs = 1

        svar_tsr = to_tensor(svar_t,use_cuda=self.use_cuda).reshape(bs,-1)
        if self.name == 'SparseQnet':
            img_t,mask_t = np.dsplit(observation,2)
            img_tsr = to_tensor(img_t,use_cuda=self.use_cuda).reshape(bs,1,self.n_frames,-1)
            mask_tsr = to_tensor(mask_t,use_cuda=self.use_cuda).reshape(bs,1,self.n_frames,-1)
            s_tsr =[img_tsr,mask_tsr,svar_tsr]
        else:
            obs_tsr = to_tensor(observation,use_cuda=self.use_cuda)#.reshape(self.batch_size,self.n_frames,-1)
            s_tsr =[obs_tsr,svar_tsr]
 
        q_tsr = self.network(s_tsr)
        a_opt = self.action_space[torch.argmax(q_tsr).item()]

        p = np.random.random() #exploration probability

        if self.is_training and p < self.epsilon:   #exploration
            action = self.random_action()
        else:                                       #exploitation
            action = a_opt
        self.epsilon = self.epsilon_decay_schedule(n=self.train_iter)
        
        self.a_t = action


        return action,a_opt
    

    def compute_loss(self,batch):
        # Sample batch
        obs_b, svar_b, a_b, r_b, t_b, next_obs_b, next_svar_b, indices, weights = batch

        #process cur state batch
        svar_tsr = to_tensor(svar_b,use_cuda=self.use_cuda)#.reshape(self.batch_size,-1)
        next_svar_tsr = to_tensor(next_svar_b,use_cuda=self.use_cuda)#.reshape(self.batch_size,-1)

        if self.name == 'SparseQnet':
            img_b,mask_b = np.dsplit(obs_b,2)
            img_tsr = to_tensor(img_b,use_cuda=self.use_cuda).reshape(self.batch_size,1,self.n_frames,-1)
            mask_tsr = to_tensor(mask_b,use_cuda=self.use_cuda).reshape(self.batch_size,1,self.n_frames,-1)
            s_tsr =[img_tsr,mask_tsr,svar_tsr]

            next_img_b,next_mask_b = np.dsplit(next_obs_b,2)
            next_img_tsr = to_tensor(next_img_b,use_cuda=self.use_cuda).reshape(self.batch_size,1,self.n_frames,-1)
            next_mask_tsr = to_tensor(next_mask_b,use_cuda=self.use_cuda).reshape(self.batch_size,1,self.n_frames,-1)
            next_s_tsr = [next_img_tsr,next_mask_tsr, next_svar_tsr]

        else:
            obs_tsr = to_tensor(obs_b,use_cuda=self.use_cuda)#.reshape(self.batch_size,self.n_frames,-1)
            s_tsr =[obs_tsr,svar_tsr]   

            next_obs_tsr = to_tensor(next_obs_b,use_cuda=self.use_cuda)#.reshape(self.batch_size,self.n_frames,-1)
            next_s_tsr = [next_obs_tsr, next_svar_tsr]
        
        #process other batchs
        a_tsr = to_longTensor(a_b,use_cuda=self.use_cuda)#.squeeze(1)#.reshape(self.batch_size,-1)
        r_tsr = to_tensor(r_b,use_cuda=self.use_cuda)#.squeeze(1)
        t_tsr = to_tensor(t_b.astype(np.float),use_cuda=self.use_cuda)#.squeeze(1)

        # compute Q for the current state
        q_val= self.network(s_tsr)

        q_val = torch.gather(q_val, 1, a_tsr.long()).squeeze(1)

        
        # compute target Q  
        with torch.no_grad():
            next_q_val = self.network(next_s_tsr)
            max_next_q_val = torch.max(next_q_val, 1)[1].unsqueeze(1)
            next_q_target_val = self.target_network(next_s_tsr)
            next_q_val = torch.gather(next_q_target_val,1, max_next_q_val).squeeze(1)

        target_q_val = r_tsr + (1 - t_tsr)*self.gamma*next_q_val


        # loss computation
        loss = self.loss(q_val,target_q_val)

        # save loss
        self.episode_loss+=loss.item()
        
        
        with torch.no_grad():
            loss_copy = loss.detach().cpu()
            weight = sum(np.multiply(weights, loss_copy))
        loss *= weight

        # compute the TD error and update the buffer
        TD_error = abs(target_q_val.detach() - q_val.detach()).cpu()
        TD_error = TD_error.numpy()
        
        #self.buffer.update_data(abs(TD_error), indices)
        self.buffer.update_priorities(indices,TD_error.reshape(-1))

        return loss

    def update_policy(self):
        if self.train_iter % self.update_frequency == 0:
            batch = self.buffer.sample(self.batch_size)
            loss = self.compute_loss(batch)
            self.network.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            #if not self.scheduler_type == 'ReduceLROnPlateau': self.scheduler_step(loss.item())

        if self.train_iter%self.sync_frequency:
            soft_update(self.target_network, self.network, self.tau)
        self.train_iter+=1

    def scheduler_step(self,quantity):
        
        if self.scheduler_type == 'LambdaLR':
            self.loss_window.append(quantity)
            if len(self.loss_window) == self.window_len:
                self.lr_scheduler.step()
        elif self.scheduler_type == 'ReduceLROnPlateau':
            self.lr_scheduler.step(quantity)
        else:
            self.lr_scheduler.step()

    def eval(self):
        self.network.eval()
        self.target_network.eval()

    def cuda(self):
        print('put in cuda')
        torch.cuda.empty_cache()
        self.network.cuda()
        self.target_network.cuda()

    def observe(self, r_t, s_t1, t_t,save=True):
        obs_t1,sVar_t1=s_t1
        if self.is_training:
            state = [np.array(self.observations),self.sVars]
            self.observations.append(obs_t1)
            self.sVars = sVar_t1
            next_state = [np.array(self.observations),self.sVars]
            if save: 
                a = tuple(self.a_t)
                #self.buffer.store(state=state, action=self.action_space.index(a),
                #              reward=r_t,done=t_t, next_state=next_state )
                self.buffer.store(
                    o = state[0],s = state[1],
                    a = self.action_space.index(a),
                    r = r_t,d = t_t,op = next_state[0],sp = next_state[1])

    def load_weights(self, output_dir):

        if output_dir is None: return
        
        ('LOAD MODEL: ',output_dir)

        self.network.load_state_dict(
            torch.load('{}/q_network.pkl'.format(output_dir))
        )
        self.target_network.load_state_dict(
            torch.load('{}/q_network.pkl'.format(output_dir))
        )

    def save_model(self,output_dir):
        print('MODEL saved in: \n',output_dir)
        torch.save(
            self.network.state_dict(),
            '{}/q_network.pkl'.format(output_dir)
        )


    def lambda_rule(self,epoch):
        if epoch%10==0:
            cur_mean = mean(self.loss_window)
            prev_mean = self.last_lambda_mean
            self.last_lambda_mean = cur_mean
            if cur_mean/prev_mean>=0.95:
                self.lr_coeff*=0.9
        return self.lr_coeff
            

    def reset_memory(self,size):
        self.buffer = PrioritizedReplayBuffer(capacity = size,
                                              o_shape = (self.n_frames,383),
                                              s_shape = (self.n_states,), 
                                              a_shape=(1,),
                                              alpha=0.8) 