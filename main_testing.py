import argparse
import os
from RL_agent.utils import *
from RL_agent.ddqn import DDQN
from utils import *
import numpy as np
from environment import Environment
from collections import namedtuple
import random
import statistics 


def eval(agent, env, seed_list, args, map = '', show=False, verbose=True):
    results = {}
    Result = namedtuple('Result',field_names=['score','n_steps','alpha_data','ending','seed'])
    agent.is_training=False

    for ep in range(args.repeats):
        step = 0
        done = False
        score=0
        alpha_data = []
        seed =None #random.randint(0,100)#random.choice(seed_list)
        env.change_seed(seed)
        
        observation = env.reset(shuffle=True,map=map)
        last_alpha = (1.0,0.0,0.0)
        state = env.get_state(observation,last_alpha)
        agent.reset(state,last_alpha)

        if verbose:print(f'\nREPEAT ({str(ep)})   - seed {seed}')

        if show:  env.maps.init_plot()
            
        if show:env.maps.plot(env.map,env.robot.mesh)
        start = time.time()

        while not done:
            cmds = env.get_cmds()
            alpha = get_classic_alpha(env.danger)
            #alpha,a_opt = agent.select_action(state)
            if verbose: print(f'STEP: {step}, Danger  = {env.danger} ,Alpha = {alpha}',end='\r',flush=True)

            env.cur_alpha = alpha
            env.target_alpha = get_classic_alpha(env.danger)
            cmd = np.sum(alpha*np.transpose(cmds),axis=-1)
            
            new_observation,reward,done = env.step(cmd)
            new_state = env.get_state(new_observation,last_alpha)

            if show:env.maps.plot(env.map,env.robot.mesh)

            if step>=args.max_iters:
                done=True
                ending =  'nothing' 
            else:
                ending = 'goal'*env.is_goal + 'collision'*env.is_coll

            #save or update stuff
            score += reward
            alpha_data.append(np.array(alpha))
            state = new_state
            last_alpha = alpha
            step+=1   

        if show: plt.close('all')
        duration = time.time()-start

        result_i = Result(score,step,alpha_data,ending,seed)
        results[ep] = result_i

        alpha_stats = get_alpha_stats(alpha_data)
        print()
        print(f' - Ending = {ending}')
        print(f' - Score = {score}')
        print(f' - N steps = {step}')
        print(f' - Alpha:')
        for i in alpha_stats.keys():
            print(f'   + {i}: {alpha_stats[i]}')

    return results


def get_alpha_stats(data):
    Stat = namedtuple('Stat',field_names=['occorrence','mean','variance'])
    stats = {}
    data = np.array(data)
    n = data.shape[0]
    modules = ['usr','ca_r','ca_t']
    for i in range(3):
        d_i = data[:,i].tolist()
        occ_i = n - d_i.count(0.0)
        mean_i = statistics.mean(d_i)
        var_i = statistics.variance(d_i)
        stats[modules[i]] = Stat(occ_i,mean_i,var_i)
    return stats



def read_results(results,modules):
    n = len(results.keys())
    tot_steps = 0
    tot_colls = 0
    tot_goals = 0
    tot_scores = 0 
    alpha1 =  {'occurrence':[], 'mean': [], 'var':[]}
    alpha2 =  {'occurrence':[], 'mean': [], 'var':[]}
    alpha3 =  {'occurrence':[], 'mean': [], 'var':[]}
    alphas = np.array([alpha1,alpha2,alpha3])

    for k in results.keys():
        r = results[k]
        tot_steps += r.n_steps
        tot_scores += r.score
        if r.ending== 'goal':
            tot_goals +=1
        elif r.ending== 'collision':
            tot_colls += 1

        stats = get_alpha_stats(r.alpha_data)
        
        for m,a in zip(modules,alphas):
            a['occurrence'].append(stats[m].occorrence/r.n_steps)  
            a['mean'].append(stats[m].mean)
            a['var'].append(stats[m].variance)
      
    mean_steps = tot_steps/n
    mean_score = tot_scores/n
    mean_alpha_res = np.zeros([3,3])
    for i in range(3):
        a = alphas[i]
        mean_alpha_res[i,0] = statistics.mean(a['occurrence'])  
        mean_alpha_res[i,1] = statistics.mean(a['mean'])
        mean_alpha_res[i,2] = statistics.mean(a['var'])

    return mean_steps,mean_score,mean_alpha_res,tot_goals,tot_colls


if __name__ == '__main__':
    print('Lets go!')
    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
    parser.add_argument('--model', default='', type=str, help='support option: train/eval')

    parser.add_argument('--dir', default='testing', type=str, help='folder to store good results')
    parser.add_argument('--parent_dir', default='/home/adriano/Desktop/RL_simulated_unicycle/', type=str, help='')
    parser.add_argument('--hidden1', default=512, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=128, type=int, help='hidden num of second fully connect layer')

    parser.add_argument('--n_seeds', default=10, type=int, help='')
    parser.add_argument('--max_iters', default=500, type=int, help='total eval test per simulation')
    parser.add_argument('--repeats', default=2, type=int, help='total eval test per simulation')


    #get the folder to store the results
    args = parser.parse_args()
    args.parent_dir = os.getcwd() 
    args.dir = os.path.join(args.parent_dir,args.dir)
    while args.model == '' or not (args.model in os.listdir(args.dir)):
        args.model = input('What is the model to test:')
    model_dir = os.path.join(args.dir,args.model)


    #load the model
    n_states = 9
    n_frames = 3
    vals = np.linspace(0.0,1.0,5)

    primitives = [(x,y,z) for x in vals for y in vals for z in vals if sum((x,y,z))==1.0]
    #primitives = [(1,0,0),(0,1,0),(0,0,1)]
    agent = DDQN(n_states, n_frames, primitives,args,is_training=False)
    print('LOAD MODEL: ',args.model)
    agent.load_weights(model_dir)
    args.agent_name = agent.name

    #get the environement
    env = Environment(args)

    seeds = random.sample(range(1, 100), args.n_seeds) 
    print(f'{args.n_seeds} seeds: {seeds}')
    #52,19,82,67,74,99,89,98,40,65,57,36,8

    results = eval(agent,env,seeds,args,show=False,map = 'labirinth',verbose=True)


    #where = os.path.join(model_dir,'results.pkl')
    #with open(where, 'wb') as handle:
    #    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #print('Results saved in: \n',where)
    modules = ['usr','ca_r','ca_t']
    mean_steps,mean_score,alphas,n_goals,n_colls = read_results(results,modules)
    print('\n\n')
    print('FINAL RESULTS')
    print('='*30)
    print(' - Mean score = ',mean_score)
    print(' - Mean steps = ',mean_steps)

    for i in range(3):
        m = modules[i]
        print(f' - {m} control:')
        print('    + Occurrence = ',round(alphas[i,0],2) )
        print('    + Mean = ',round(alphas[i,1],2))
        print('    + Variance = ',round(alphas[i,2],2))

    print(' - N goals = ',n_goals)
    print(' - N colls = ',n_colls)

