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
from copy import deepcopy


def train(agent,env,args,verbose=True):
    shuffle=True
    model_dir = args.output_weight # get_folder(name=args.output_weight,**kargs)
    train_plot = args.output_plot # get_folder(name=args.output_plot,**kargs)
    best_score = 1000

    agent = warmup(agent,env,
                   n_steps=args.warmup,
                   random_wu = args.random_warmup,
                   shuffle=shuffle)
    
    try:
        epoch = args.output_plot.epochs[-1]
    except:
        epoch=0

    print('\n\nStart TRAINING\n','-'*20)

    while epoch<args.max_epochs:
        agent.is_training=True
        step = 0
        done = False
        observation = env.reset(map = 'random',shuffle=shuffle)
        last_alpha = (1.0,0.0,0.0)
        episode_reward=0
        state = env.get_state(observation,last_alpha)
        agent.reset(state,last_alpha)

        print(f'\nEPOCH: {epoch}')
        if verbose: print('-'*20+'\n'*6)

        start = time.time()
        while not done:

            alpha,a_opt = agent.select_action(state)

            header = 'STEP: ' + str(step) +' (train)' 
            env.update_alpha(alpha,get_classic_alpha(env.danger))

            cmds = env.get_cmds()
            cmd = np.sum(alpha*np.transpose(cmds),axis=-1)
            if env.collision_forecast(*cmd):
                cmd  = cmds[1]


            new_observation,reward,done = env.step(cmd)
            new_state = env.get_state(new_observation,last_alpha)
            agent.observe(reward,new_state, done)

            agent.update_policy()
            #agent_ = deepcopy(agent)
            #env_ = deepcopy(env)
            #score,best_score = cross_validation(agent_,env_,model_dir,best_score,times=5)
            #score= cross_validation(agent_,env_,times=2,verbose=True)
            #if score>best_score:
            #    best_score = score
            #    dir = os.path.join(model_dir,'score_'+str(int(score)))
            #    os.makedirs(dir, exist_ok=True)
            #    agent.save_model(dir)
            #    #agent.scheduler_step(score)
            
            if verbose:write_console(header,alpha, a_opt,env.danger,agent.get_lr(),env.dt)

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
        train_plot.store(epoch,episode_reward,loss,agent.lr,agent.epsilon)
        train_plot.save_dict()
        train_plot.plot_and_save()

        display_results(step,agent,result,episode_reward,loss, time.time()-start,train=True)

        if True: #epoch>10:
            score= cross_validation(agent,env,times=2,verbose=True)
            if score>best_score:
                best_score = score
                dir = os.path.join(model_dir,'score_'+str(int(score)))
                os.makedirs(dir, exist_ok=True)
                agent.save_model(dir)
                agent.scheduler_step(score)
        epoch+=1


def eval(agent,env, map = None,mode='classic',shuffle=True, show=True,verbose=True):
    score = []
    agent.is_training=False
    n_episodes = len(map)
    for ep in range(n_episodes):
        if verbose:print('-'*5 + ' EVALUATION '+str(ep)+'-'*5)#+'\n'*6)
        step = 0
        done = False
        episode_reward=0
        
        observation = env.reset(shuffle,map=map[ep])
        last_alpha = (1.0,0.0,0.0)
        state = env.get_state(observation,last_alpha)
        agent.reset(state,last_alpha)

        if show:  env.maps.init_plot()
            
        if show:env.maps.plot(env.map,env.robot.mesh)
        start = time.time()

        while not done:
            cmds = env.get_cmds()
            
            if mode=='classic':
                alpha = get_classic_alpha(env.danger)
                #if verbose: print(f'STEP: {step}, Danger  = {env.danger} ,Alpha = {alpha}',end='\r',flush=True)
            else:
                alpha,a_opt = agent.select_action(state)
                if verbose: print(f'STEP: {step}, Danger  = {env.danger} ,Alpha = {alpha}',end='\r',flush=True)
            env.cur_alpha = alpha
            env.target_alpha = get_classic_alpha(env.danger)
            cmd = np.sum(alpha*np.transpose(cmds),axis=-1)
            
            new_observation,reward,done = env.step(cmd)
            new_state = env.get_state(new_observation,last_alpha)

            if show:env.maps.plot(env.map,env.robot.mesh)
                #plot(env.goal,env.obs_mesh,env.check_points,env.robot.mesh,env.dt)

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
        #score_i = (args.max_episode_length-step)*env.is_goal
        score.append(episode_reward)
        if show: plt.close('all')
        duration = time.time()-start
        if verbose:display_results(step,agent,result,score[-1],'-',duration,train=False)
    
    return score


def cross_validation(agent,env,times=5,verbose=False):
    if verbose:print('VALIDATION')
    score_model = eval(agent,env,map=[None]*times,mode='random',
                        shuffle=True,show=False,verbose=False)
    score = mean(score_model)
    if verbose:print('SCORE: ', score)
    
    return score
    

def warmup(agent,env,n_steps:int,random_wu = True,shuffle=True):
    is_warmup = True
    step_warmap=1
    observation = env.reset(shuffle=shuffle)
    last_alpha = (1.0,0.0,0.0)
    state = env.get_state(observation,last_alpha)
    agent.reset(state,last_alpha)
    while is_warmup:
        step = 0
        done = False
        observation = env.reset(shuffle=shuffle)
        last_alpha = (1.0,0.0,0.0)
        state = env.get_state(observation,last_alpha)
        agent.reset(state,last_alpha)
        while not done:
            print(f'WARMUP {step_warmap} (mem. {round(agent.buffer.get_capacity()*100,1)}%)',end='\r',flush=True)
            if random_wu:
                alpha = agent.random_action()
            else:
                alpha,_ = agent.select_action(state)

            env.update_alpha(alpha,get_classic_alpha(env.danger))
            cmds = env.get_cmds()
            cmd = np.sum(alpha*np.transpose(cmds),axis=-1)
            new_observation,reward,done = env.step(cmd)

            new_state = env.get_state(new_observation,last_alpha)
            agent.observe(reward,new_state, done)
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

    for mode in ['eval','classic']: # ['eval_'+str(i) for i in range(10)]:
        step = 0
        done = False
        observation = env.reset(map=map)
        last_alpha = (1.0,0.0,0.0)
        state = env.get_state(observation,last_alpha)
        agent.reset(state,last_alpha)
        env.maps.init_plot()
        env.maps.plot(env.map,env.robot.mesh)
        image_folder = os.path.join(dir,'images_'+mode)
        os.makedirs(image_folder, exist_ok=True)

        start = time.time()
        while not done:
            cmds = env.get_cmds()
            
            if mode=='classic':
                alpha = get_classic_alpha(env.danger)
                if verbose: print(f'STEP: {step}, Danger  = {env.danger} ,Alpha = {alpha}',end='\r',flush=True)
            else:
                alpha,a_opt = agent.select_action(state)
                if verbose: print(f'STEP: {step}, Danger  = {env.danger} ,Alpha = {alpha}',end='\r',flush=True)

            env.cur_alpha = alpha
            env.target_alpha = get_classic_alpha(env.danger)
            cmd = np.sum(alpha*np.transpose(cmds),axis=-1)
            
            new_observation,reward,done = env.step(cmd)
            new_state = env.get_state(new_observation,last_alpha)
            
            env.maps.plot(env.map,env.robot.mesh)

            img_name = 'step_'+ str(step).zfill(3) +'.png'
            path = os.path.join(image_folder,img_name)
            plt.savefig(path)

            if step>=args.max_episode_length:
                done=True
                result =  'Nothing' 
            else:
                result = 'Goal riched'*env.is_goal + 'Collision'*env.is_coll

            #save or update stuff
            state = new_state
            last_alpha = alpha
            step+=1        
        
        video_name = 'video_'+str(mode)
        make_video(image_folder,video_folder,video_name,dt = env.dt)
    
    duration = time.time()-start
    if verbose:display_results(step,agent,result,score[-1],'-',duration,train=False)

    return score






if __name__ == '__main__':
    print('Lets go!')
    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
    parser.add_argument('--mode', default='train', type=str, help='support option: train/eval')
    parser.add_argument('--model', default='', type=str, help='support option: train/eval')

    parser.add_argument('--env', default='static_DDDQN', type=str, help='open-ai gym environment')
    parser.add_argument('--n_obs', default=5, type=str, help='number of obstacles in the map')

    parser.add_argument('--hidden1', default=512, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=128, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--warmup', default=500, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--bsize', default=256, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=2000, type=int, help='memory size')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=1000, type=int, help='')
    parser.add_argument('--discount', default=0.999, type=float, help='')

    parser.add_argument('--output_weight', default='weights', type=str, help='folder to save the weigths')
    parser.add_argument('--output_plot', default='plots', type=str, help='folder to save the plots')
    parser.add_argument('--good_results', default='good_results', type=str, help='folder to store good results')

    parser.add_argument('--parent_dir', default='/home/adriano/Desktop/RL_simulated_unicycle/', type=str, help='')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--max_train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=1.0, type=float, help='linear decay of exploration policy')
    parser.add_argument('--epsilon_decay', default=0.9995, type=float, help='linear decay of exploration policy')
    parser.add_argument('--lambda_window', default=100, type=int, help='linear decay of exploration policy')

    parser.add_argument('--max_epochs', default=300, type=int, help='total train epochs')
    parser.add_argument('--n_frames', default=3, type=int, help='train iters each timestep')
    parser.add_argument('--random_warmup', default=True, type=int, help='')

    args = parser.parse_args()
    args.parent_dir = os.getcwd() 

    
        
   
    
    #BEST RESULTS
    #args.fixed_pos =[
    #    [[0.3,0.75],[-0.5,1.5]],
    #    [[0.5,1.5],[-0.3,0.75]]
    #]


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
    args.output_plot = TrainData(parent_dir=plot_dir,env = args.env,name=agent.name) 

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
        train(agent,env,args,verbose=True)
    elif args.mode == 'rec':
        video_dir = get_folder(name='videos',**kargs)
        score = record_video(agent,env, map = 'labirinth',dir=video_dir)
    elif args.mode == 'test':
        results_dir = get_folder(name='videos',**kargs)
    else:
        print(args.mode)
        score = eval(agent,env,mode=args.mode,shuffle=True,show=True,map = ['random']*10,verbose=True)


