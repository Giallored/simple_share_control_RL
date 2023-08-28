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


def eval(agent, env, args,r_tuple,map = '', show=False, verbose=True,modes='eval'):
    results_eval = {}
    results_classic = {}

    agent.is_training=False
    shuffle = True

    for ep in range(args.repeats):
        for mode in modes:
            step = 0
            done = False
            score=0
            alpha_data = []
            seed =None
            env.change_seed(seed)
            shuffle = True if mode == 'eval' else False
            observation,last_alpha = env.reset(shuffle,map=map)
            state = env.get_state(observation,last_alpha)
            agent.reset(state,last_alpha)

            print(f'REPEAT ({str(ep)}) - mode = {mode}',end='\r',flush=True)

            if show:  env.maps.init_plot()
                
            if show:env.maps.plot(env.map,env.robot.mesh)
            start = time.time()

            while not done:
                cmds = env.get_cmds()
                if env.danger<=2: alpha=(1.0,0.0,0.0)
                else:
                    if mode=='classic':
                        alpha = get_classic_alpha(env.danger)
                    else:
                        alpha,a_opt = agent.select_action(state)
                if verbose: print(f'STEP: {step}, Danger  = {env.danger} ,Alpha = {alpha}',end='\r',flush=True)

                env.cur_alpha = alpha
                env.target_alpha = get_classic_alpha(env.danger)
                cmd = np.sum(alpha*np.transpose(cmds),axis=-1)
                
                new_observation,reward,done = env.step(cmd,alpha)
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

            result_i = r_tuple(score,step,alpha_data,ending,seed)
            if mode =='classic':
                results_classic[ep] = result_i
            else:
                results_eval[ep] = result_i

            alpha_stats = get_alpha_stats(alpha_data)
            if verbose:
                print()
                print(f' - Ending = {ending}')
                print(f' - Score = {score}')
                print(f' - N steps = {step}')
                print(f' - Alpha:')
                for i in alpha_stats.keys():
                    print(f'   + {i}: {alpha_stats[i]}')

    return results_eval,results_classic


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
        
        if r.ending== 'goal':
            tot_goals +=1
            tot_scores += r.score
        elif r.ending== 'collision':
            tot_colls += 1

        stats = get_alpha_stats(r.alpha_data)
        
        for m,a in zip(modules,alphas):
            a['occurrence'].append(stats[m].occorrence/r.n_steps)  
            a['mean'].append(stats[m].mean)
            a['var'].append(stats[m].variance)
      
    mean_steps = tot_steps/n
    mean_score = tot_scores/tot_goals
    mean_alpha_res = np.zeros([3,3])
    for i in range(3):
        a = alphas[i]
        mean_alpha_res[i,0] = statistics.mean(a['occurrence'])  
        mean_alpha_res[i,1] = statistics.mean(a['mean'])
        mean_alpha_res[i,2] = statistics.mean(a['var'])

    return mean_steps,mean_score,mean_alpha_res,tot_goals,tot_colls


def show_results(results_dict,modules,n_repeats,f,mode,verbose = True):
    mean_steps,mean_score,alphas,n_goals,n_colls = read_results(results_dict,modules)  

    if verbose:
        print('\n')
        print(f'FINAL RESULTS ({mode})',mode)
        print('-'*30)
        print(' - Mean score = ',mean_score)
        print(' - Mean steps = ',mean_steps)

    f.write('\nMODE =%s\r\n'%mode)
    f.write(' - Mean score = %f\r\n'%mean_score)
    f.write(' - Mean steps = %f\r\n'%mean_steps)

    for i in range(3):
        m = modules[i]
        f.write(' - %s control:\r'%m)
        f.write('    + Occurrence = %f\r'%round(alphas[i,0],2))
        f.write('    + Mean = %f\r'%round(alphas[i,1],2))
        f.write('    + Variance = %f\r\n'%round(alphas[i,2],2))

        if verbose: 
            print(f' - {m} control:')
            print('    + Occurrence = ',round(alphas[i,0],2) )
            print('    + Mean = ',round(alphas[i,1],2))
            print('    + Variance = ',round(alphas[i,2],2))
    
    f.write(' - goals = %f\r\n'%(n_goals/n_repeats))
    f.write(' - colls = %f\r\n'%(n_colls/n_repeats))

    if verbose: 
        print(' - goals = ',round((n_goals/n_repeats*100),3), '%')
        print(' - colls = ',round((n_colls/n_repeats*100),3), '%')
    
    return n_goals/n_repeats*100
 




if __name__ == '__main__':
    print('Lets go!')
    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
    parser.add_argument('--model', default='', type=str, help='support option: train/eval')
    #parser.add_argument('--modes', default=['classic'], type=list, help='support option: ["eval","classic"]')
    parser.add_argument('--test_model', default='True', type=bool, help='')
    parser.add_argument('--test_classic', default='0', type=bool, help='')

    parser.add_argument('--dir', default='testing', type=str, help='folder to store good results')
    parser.add_argument('--hidden1', default=512, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=128, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--n_obs', default=5, type=int, help='n of obstacles')

    parser.add_argument('--n_seeds', default=10, type=int, help='')
    parser.add_argument('--max_iters', default=200, type=int, help='max n step per simulation')
    parser.add_argument('--repeats', default=100, type=int, help='total eval test per simulation')


    #get the folder to store the results
    args = parser.parse_args()
    args.parent_dir = os.getcwd() 
    args.dir = os.path.join(args.parent_dir,args.dir)
    #args.modes = ['eval'*args.test_model,'classic'*args.test_classic]
    args.modes = ['eval']
    while args.model == '':# or not (args.model in os.listdir(args.dir)):
        args.model = input('What is the model to test:')
    model_dir = os.path.join(args.dir,args.model)

    report_name = os.path.join(model_dir,'report.txt')
    f_r = open(report_name,"w+")
    f_r.write('SUCCESS RATES OF ALL IMPORTANT RUNS\n'+'-'*20)
    model_list = sorted(os.listdir(model_dir),reverse=True)
    print(f'There are {len(model_list)} models to test!')

    #load the model
    n_states = 9
    n_frames = 3
    vals = np.linspace(0.0,1.0,5)
    #primitives = [(x,y,z) for x in vals for y in vals for z in vals if sum((x,y,z))==1.0]
    primitives = [(1,0,0),(0,1,0),(0,0,1)]
    agent = DDQN(n_states, n_frames, primitives,args,is_training=False)
    args.agent_name = agent.name

    #get the environement
    env = Environment(args)
    seeds = random.sample(range(1, 100), args.n_seeds) 
    #print(f'{args.n_seeds} seeds: {seeds}')
    Result = namedtuple('Result',field_names=['score','n_steps','alpha_data','ending','seed'])

    modules = ['usr','ca_r','ca_t']

    for model_name in model_list:
        if model_name == 'scheduled' or model_name =='report.txt' :
            continue
        model2load_dir = os.path.join(model_dir,model_name)
        print('\n\n'+'='*20)
        print('Mode: ',args.modes)
        print('Model: ',model_name,' from ', args.model)
        agent.load_weights(model2load_dir)

        results_model, results_classic= eval(agent,env,args,Result,show=False,map = 'random',verbose=False,modes=args.modes)
        
        file_name = os.path.join(model2load_dir,"result.txt")
        f= open(file_name,"w+")
        f.write('PARAMS:\n * seed = None (uses clock)\n * %d iterations\n * max n. steps = 200\n * 0.1 random action by the user\r\n'%args.repeats)
        if 'eval' in args.modes: success_rate = show_results(results_model,modules,int(args.repeats),f,mode='eval',verbose=True)
        if 'classic' in args.modes: _ = show_results(results_classic,modules,int(args.repeats),f,mode='classic',verbose=False)
        f.close()
        f_r.write(model_name + ' : ' + str(success_rate) + '\n')
        print('='*20)
    f_r.close()




#,results_classic 

