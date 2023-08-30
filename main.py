import numpy as np
from environment import Environment
import matplotlib.pyplot as plt
import random
import time
import matplotlib.animation as animation
from RL_agent.utils import *
from RL_agent.ddqn import DDQN

from utils import *
import argparse
import os
from statistics import mean
import wandb

def train(agent,env,args,verbose=True):
    shuffle=True
    
    model_dir = args.output_weight 
    score_th = 1000
    epoch=0
    map = 'random'

    #prefill the memory with some iterations
    agent = warmup(agent,env,map=map,
                n_steps=args.warmup,
                random_wu = args.random_warmup,
                shuffle=shuffle)

    print('\n\nStart TRAINING\n','-'*20)

    while epoch<args.max_epochs:
        agent.is_training=True
        #init vars
        step = 0
        done = False
        episode_reward=0

        #reset all 
        observation,last_alpha = env.reset(map = map,shuffle=shuffle)
        state = env.get_state(observation,last_alpha)
        agent.reset(state,last_alpha)

        print(f'\nEPOCH: {epoch}')
        if verbose: print('-'*20+'\n'*6)

        start = time.time()
        
        while not done:
            cmds = env.get_cmds()
            alpha,a_opt = agent.select_action(state)
            cmd = np.sum(alpha*np.transpose(cmds),axis=-1)
            new_observation,reward,done = env.step(cmd,alpha)
            new_state = env.get_state(new_observation,last_alpha)

            exp_alpha = get_classic_alpha(env.danger) # If you want to integrate some LfD
            agent.observe(reward,new_state, done,exp_alpha)

            agent.update_policy()
            lr = agent.get_lr()

            if verbose:
                header = 'STEP: ' + str(step) +' (train)' 
                write_console(header,alpha, a_opt,env.danger,lr,env.dt)

            if step>=args.max_episode_length:
                done=True
                loss = agent.episode_loss/step
                result =  'Nothing' 
            else:
                result = 'Goal riched'*env.is_goal + 'Collision'*env.is_coll

            #save or update stuff
            episode_reward += reward
            state = new_state
            last_alpha = alpha
            step+=1
       
        agent.save_model(model_dir)
        loss = agent.episode_loss/step
        display_results(step,agent,result,episode_reward,loss, time.time()-start,train=True)

        if epoch>10:
            score,goals= cross_validation(agent,env,times=10,verbose=True)
            if score>= score_th:#goals>=0.9:
                dir = os.path.join(model_dir,'g_'+str(round(goals*100))+'_s'+str(int(score))+'_e'+str(epoch))
                os.makedirs(dir, exist_ok=True)
                agent.save_model(dir)
                agent.scheduler_step(score)
        else:
            score = 0
        wandb.log({"eval_score": score, "loss": loss, "lr":lr,"eps":agent.epsilon})
        epoch+=1
    wandb.finish()


def eval(agent,env, map = 'random',n=10,mode='classic',shuffle=True, show=True,verbose=False):
    score = []
    agent.is_training=False
    goals=0
    for ep in range(n):
        if verbose:print('-'*5 + ' EVALUATION '+str(ep)+'-'*5)#+'\n'*6)
        step = 0
        done = False
        episode_reward=0
        if show:  env.maps.init_plot()
        observation,last_alpha = env.reset(shuffle,map=map)
        state = env.get_state(observation,last_alpha)
        agent.reset(state,last_alpha)
            
        start = time.time()

        while not done:
            #get state commands
            cmds = env.get_cmds()

            #get blending vars
            if env.danger<=2: alpha=(1.0,0.0,0.0)
            else:
                if mode=='classic':
                    alpha = get_classic_alpha(env.danger)
                else:
                    alpha,a_opt = agent.select_action(state)
            if verbose: print(f'STEP: {step}, Danger  = {env.danger} ,Alpha = {alpha}',end='\r',flush=True)
            
            #blend commands to collect the final command
            cmd = np.sum(alpha*np.transpose(cmds),axis=-1)
            
            #make the step and collect the new state
            new_observation,reward,done = env.step(cmd,alpha)
            new_state = env.get_state(new_observation,last_alpha)

            if show:env.maps.plot(env.map,env.robot.mesh,alpha)

            if step>=args.max_episode_length:
                done=True
                result =  'Nothing' 
            else:
                result = 'Goal riched'*env.is_goal + 'Collision'*env.is_coll

            #save or update stuff
            episode_reward += reward
            state = new_state
            last_alpha = alpha
            step+=1        

        score.append(episode_reward)
        if env.is_goal:
            goals+=1
        if show: plt.close('all')
        duration = time.time()-start
        if verbose:display_results(step,agent,result,score[-1],'-',duration,train=False)
    
    return score,goals


def cross_validation(agent,env,times=5,verbose=False):
    if verbose:print('VALIDATION')
    score_model,goals = eval(agent,env,map=None,n=times,mode='random',
                        shuffle=True,show=False,verbose=False)
    score = mean(score_model)
    goals = goals/times
    if verbose:print('SCORE: ', score)
    
    return score,goals


def fill_ERB(agent,env): 
    while not agent.buffer.full:
        step = 0
        done = False
        observation,last_alpha = env.reset()
         
        state = env.get_state(observation,last_alpha)
        agent.reset(state,last_alpha)
        while not done:
            print(f'FILLING ERB (mem. {round(agent.buffer.get_capacity()*100,1)}%)',end='\r',flush=True)
            cmds = env.get_cmds()
            alpha = get_classic_alpha(env.danger)
            cmd = np.sum(alpha*np.transpose(cmds),axis=-1)
            new_observation,reward,done = env.step(cmd,alpha)
            new_state = env.get_state(new_observation,last_alpha)
            exp_alpha = get_classic_alpha(env.danger)
            agent.observe(reward,new_state, done,exp_alpha)
            step+=1


def warmup(agent,env,map,n_steps:int,random_wu = True,shuffle=True):
    is_warmup = True
    step_warmap=1
     
    while is_warmup:
        step = 0
        done = False
        observation,last_alpha = env.reset(map=map,shuffle=shuffle)
        state = env.get_state(observation,last_alpha)
        agent.reset(state,last_alpha)
        while not done:
            print(f'WARMUP {step_warmap} (mem. {round(agent.buffer.get_capacity()*100,1)}%)',end='\r',flush=True)
            cmds = env.get_cmds()
            if random_wu:
                alpha = agent.random_action()
            else:
                alpha,_ = agent.select_action(state)
            cmd = np.sum(alpha*np.transpose(cmds),axis=-1)
            new_observation,reward,done = env.step(cmd,alpha)

            new_state = env.get_state(new_observation,last_alpha)
            exp_alpha = get_classic_alpha(env.danger)
            agent.observe(reward,new_state, done,exp_alpha)
            is_warmup = step_warmap<n_steps
            if not is_warmup:
                break
            else:
                step_warmap+=1
            step+=1
    return agent


def record_video(agent,env, dir, map = None,verbose = False):
    score = []
    agent.is_training=False
    video_folder = os.path.join(dir,'video')
    os.makedirs(video_folder, exist_ok=True)

    for mode in ['eval_'+str(i) for i in range(10)]: # ['eval','classic']: 
        step = 0
        done = False
        observation,last_alpha = env.reset(map=map)
         
        state = env.get_state(observation,last_alpha)
        agent.reset(state,last_alpha)
        env.maps.init_plot()
        
        image_folder = os.path.join(dir,'images_'+mode)
        os.makedirs(image_folder, exist_ok=True)

        start = time.time()
        while not done:
            cmds = env.get_cmds()
            
            if env.danger<=2: alpha=(1.0,0.0,0.0)
            else:
                if mode=='classic':
                    alpha = get_classic_alpha(env.danger)
                else:
                    alpha,a_opt = agent.select_action(state)
            env.cur_alpha = alpha
            cmd = np.sum(alpha*np.transpose(cmds),axis=-1)
            
            new_observation,reward,done = env.step(cmd,alpha)
            new_state = env.get_state(new_observation,last_alpha)
            
            env.maps.plot(env.map,env.robot.mesh,alpha)

            img_name = 'step_'+ str(step).zfill(3) +'.png'
            path = os.path.join(image_folder,img_name)
            plt.savefig(path)

            if step>=args.max_episode_length:
                done=True
                result =  'Nothing' 
            else:
                result = 'Goal riched'*env.is_goal + 'Collision'*env.is_coll

            #save or update sepsilontuff
            state = new_state
            last_alpha = alpha
            step+=1        
        
        video_name = 'video_'+str(mode)
        make_video(image_folder,video_folder,video_name,dt = env.dt)
        plt.close('all')
    
    duration = time.time()-start
    if verbose:display_results(step,agent,result,score[-1],'-',duration,train=False)

    return score






if __name__ == '__main__':
    print('Lets go!')
    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
    parser.add_argument('--mode', default='train', type=str, help='support option: train/eval')
    parser.add_argument('--model', default='', type=str, help='support option: train/eval')

    parser.add_argument('--env', default='static_DDQN', type=str, help='open-ai gym environment')
    parser.add_argument('--n_obs', default=5, type=str, help='number of obstacles in the map')

    parser.add_argument('--hidden1', default=512, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=128, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--warmup', default=500, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--bsize', default=256, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=1000, type=int, help='memory size')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--max_episode_length', default=200, type=int, help='')
    parser.add_argument('--validate_steps', default=1000, type=int, help='')
    parser.add_argument('--discount', default=0.999, type=float, help='')

    parser.add_argument('--output_weight', default='weights', type=str, help='folder to save the weigths')
    parser.add_argument('--output_plot', default='plots', type=str, help='folder to save the plots')
    parser.add_argument('--good_results', default='good_results', type=str, help='folder to store good results')

    parser.add_argument('--parent_dir', default='/home/adriano/Desktop/RL_simulated_unicycle/', type=str, help='')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--max_train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=1.0, type=float, help='linear decay of exploration policy')
    parser.add_argument('--epsilon_decay', default=0.9999, type=float, help='linear decay of exploration policy')

    parser.add_argument('--max_epochs', default=500, type=int, help='total train epochs')
    parser.add_argument('--n_frames', default=3, type=int, help='train iters each timestep')
    parser.add_argument('--random_warmup', default=True, type=int, help='')

    args = parser.parse_args()
    args.parent_dir = os.getcwd() 


    #get folders
    kargs = {
        'env_name':args.env,
        'parent_dir':args.parent_dir,
        'mode':'train',
        'model2load':args.model
    }

    model_dir = get_folder(name=args.output_weight,**kargs)
    plot_dir = get_folder(name=args.output_plot,**kargs)

    #instatiate the agent
    n_states = 9
    vals = np.linspace(0.0,1.0,5)
    #primitives = [(x,y,z) for x in vals for y in vals for z in vals if sum((x,y,z))==1.0]
    primitives = [(1,0,0),(0,1,0),(0,0,1)]
    agent = DDQN(n_states, args.n_frames, primitives,args)

    args.output_weight = model_dir

    #load model (if necessary)
    if not args.model =="":
        print('LOAD MODEL: ',args.model, ' for ', args.mode)
        agent.load_weights(model_dir)
        if args.mode == 'train':
            args.random_warmup = False
            where = os.path.join(plot_dir,'train_dict.pkl')
            with open(where, 'rb') as handle:
                dict = pickle.load(handle)
            args.output_plot.load_dict(dict)
            agent.set_hp(args.output_plot) 
    args.agent_name = agent.name
    #get the environement
    env = Environment(args)

    if args.mode == 'train':
        wandb.init(project="SharedControlRL",
        config={
        "learning_rate": agent.lr,
        "agent": 'DDQN',#agent.name,
        "environment": "sintetic",
        "map":'random_'+str(args.n_obs)+'obs',
        "model_dir":model_dir,
        "epochs": args.max_epochs,
        "episode_lenght":args.max_episode_length,
        "epsilon":args.epsilon,
        "epsilon_decay":args.epsilon_decay,
        "n_frames":args.n_frames,
        "n_primitives":len(primitives),
        "n_state":n_states,
        "warmup_episodes":args.warmup,
        "ERB_size":args.rmsize,
        "batch":args.bsize,

        }
    )
        WANDB_MODE="offline"
        train(agent,env,args,verbose=True)
    elif args.mode == 'rec':
        WANDB_MODE="offline"
        video_dir = get_folder(name='videos',**kargs)
        score = record_video(agent,env, map = 'labirinth',dir=video_dir)
    #elif args.mode == 'test':
    #    WANDB_MODE="offline"
    #    score = eval(agent,env,mode=args.mode,shuffle=True,show=True,map = ['random']*10,verbose=True)
    else:
        WANDB_MODE="offline"
        print(args.mode)
        score,goals = eval(agent,env,map = 'random',n=10,mode=args.mode,shuffle=True,show=True,verbose=True)


