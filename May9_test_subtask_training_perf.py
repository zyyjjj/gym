#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 12:10:36 2021

@author: yujiazhang
"""

import os, sys
import gym
import numpy as np
import matplotlib.pyplot as plt
import warnings
from stable_baselines3 import A2C, SAC, PPO, TD3, HER, DQN
from humanoidstandup import HumanoidStandupEnv
from Evaluate_training_performance import EvalCallback, evaluate_policy

module_path = os.path.abspath(os.path.join('.'))
# if module_path not in sys.path:
sys.path.append(module_path + "/gym/envs/classic_control")
sys.path.append(module_path + "/gym/envs/mujoco")
print('{}'.format(sys.path))

def run_training_iterations(eval_env, model, n_iterations, learning_steps, eval_episodes, sim_steps_upper_bound):
    rewards = np.zeros((n_iterations, 2))
    infos = []
    
    for n_iter in range(n_iterations):
        model.learn(total_timesteps = learning_steps)
        episode_rewards, episode_lengths, episode_infos, info_names = \
            evaluate_policy(model, eval_env, 
                            n_eval_episodes = eval_episodes, 
                            sim_steps_upper_bound = sim_steps_upper_bound ,
                            return_episode_rewards = True)
    
        reward_mean, reward_std = np.mean(episode_rewards), np.std(episode_rewards)
        episode_infos_array = np.stack(episode_infos)
        episode_infos_mean, episode_infos_std = np.mean(episode_infos_array, axis=0), np.std(episode_infos_array, axis=0)
        
        rewards[n_iter] = np.array([reward_mean, reward_std])
        infos.append([episode_infos_mean, episode_infos_std])

    return rewards, infos



def process_output(rewards, infos):
    main_rewards = rewards.transpose()[0]
    main_rewards_sd = rewards.transpose()[1]
        
    subtask_1_rewards = [infos[i][0][3] for i in range(len(infos))]
    subtask_1_rewards_sd = [infos[i][1][3] for i in range(len(infos))]
    
    subtask_2_rewards = [infos[i][0][4] for i in range(len(infos))]
    subtask_2_rewards_sd = [infos[i][1][4] for i in range(len(infos))]
    
    #control_costs = [infos[i][0][1] for i in range(len(infos))]
    #control_costs_sd = [infos[i][1][1] for i in range(len(infos))]
    
    return main_rewards, main_rewards_sd, subtask_1_rewards, subtask_1_rewards_sd, subtask_2_rewards, subtask_2_rewards_sd 
    


def plot_two_scales(data1, data2_list, label1, label2_list, title, data1_sd=None, data2_sd_list=None):
    #assert(len(data1) == len(data2))
    
    colors = ['red', 'blue', 'orange', 'green']
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(data1, color = colors[0], label=label1)
    ax.set_ylabel(label1)
    
    datasize = len(data1)
    
    if data1_sd is not None:
        ax.fill_between(range(datasize), [data1[j]-1.96*data1_sd[j] for j in range(datasize)],\
                [data1[j]+1.96*data1_sd[j] for j in range(datasize)],
                  color = colors[0], alpha=0.1)

    ax2 = ax.twinx()
    for i in range(len(data2_list)):
        ax2.plot(data2_list[i], color = colors[i+1], label=label2_list[i])
        if data2_sd_list[i] is not None:
            #print(i, 'sd not none')
            #print([data2_list[i][j]-1.96*data2_sd_list[i][j] for j in range(datasize)])
            ax2.fill_between(range(datasize), [data2_list[i][j]-1.96*data2_sd_list[i][j] for j in range(datasize)],\
                [data2_list[i][j]+1.96*data2_sd_list[i][j] for j in range(datasize)],
                  color = colors[i+1], alpha=0.1)
    ax2.set_ylabel(label2_list)

    ax.legend(loc = 'upper left')
    ax2.legend(loc = 'upper right')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    eval_env = HumanoidStandupEnv()
    model = model = A2C('MlpPolicy', eval_env, verbose = 1)
    n_iterations = 100
    learning_steps = 1000
    eval_episodes = 10
    sim_steps_upper_bound = 1000
    
    rewards, infos = run_training_iterations(eval_env, model, n_iterations, learning_steps, eval_episodes, sim_steps_upper_bound)
    
    main_rewards, main_rewards_sd, _, _, subtask_2_rewards, subtask_2_rewards_sd = process_output(rewards, infos)
    
    plot_two_scales(main_rewards, [subtask_2_rewards], 'reward', ['abdomen vertical velocity'], 
               title = 'Main reward and root vertical velocity over training iterations',
               data1_sd = main_rewards_sd, data2_sd_list = [subtask_2_rewards_sd])


    
    
    