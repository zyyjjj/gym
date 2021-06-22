#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:36:39 2021

@author: yujiazhang
"""
__all__ = ["Monitor", "ResultsWriter", "get_monitor_files", "load_results"]

import csv
import json
import os, sys
from glob import glob
from typing import Dict, List, Optional, Tuple, Union, Callable

sys.path.append('~/.local/lib/python3.7/site-packages')
import pybullet_envs
import dill
import wandb

import pdb

import gym
import numpy as np
import pandas
import warnings
import time
import argparse
import itertools
import random

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common import logger
from stable_baselines3 import A2C, SAC, PPO, TD3
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv

#if typing.TYPE_CHECKING:
#    from stable_baselines.common.base_class import BaseRLModel

def evaluate_policy(
    model,#: "BaseRLModel",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    sim_steps_upper_bound: int = 1000,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (Optional[float]) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    if hasattr(model.policy, 'recurrent'):
        is_recurrent = model.policy.recurrent
    else:
        is_recurrent = False

    episode_rewards, episode_lengths, episode_infos = [], [], []

    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
            # Because recurrent policies need the same observation space during training and evaluation, we need to pad
            # observation to match training shape. See https://github.com/hill-a/stable-baselines/issues/1015
            if is_recurrent:
                zero_completed_obs = np.zeros((model.n_envs,) + model.observation_space.shape)
                zero_completed_obs[0, :] = obs
                obs = zero_completed_obs
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        episode_info = []
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            new_obs, reward, done, _info = env.step(action)
            if episode_length == 0:
                info_names = list(_info.keys())
                num_subtasks = len(list(_info.values()))
            episode_info.append(list(_info.values())[:num_subtasks])
            if is_recurrent:
                obs[0, :] = new_obs
            else:
                obs = new_obs
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
            
            done = bool(episode_length > sim_steps_upper_bound)
        
        #pdb.set_trace()
        
        # info_names = list(_info.keys())
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        #episode_infos.append(episode_info)
        episode_infos.append(np.sum(np.stack(episode_info)[-50:,],axis=0))
        

                
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: {:.2f} < {:.2f}".format(mean_reward, reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths, np.array(episode_infos), info_names

    return mean_reward, std_reward

# function for parsing arguments passed to the hyperparameters

def parse_hp_args():
    parser = argparse.ArgumentParser(description='input names and values of hyperparameters to vary')
    parser.add_argument('-a', '--algorithm', choices = ['A2C', 'PPO', 'SAC', 'DQN'], help='StableBaselines3 RL algorithms to run')
    parser.add_argument('-e', '--env', default='HumanoidBulletEnv-ST-v0', help='environment to evaluate the algorithm')
    parser.add_argument('-p', '--param-to-vary', action='append', 
            help='names of hyperparameter(s) to be varied', required=True)
    parser.add_argument('-v', '--values', required=True, nargs='+', action='append',
            help='values of the varying hyperparameter(s)')
    parser.add_argument('-o', '--outputdir', help='directory to store output')
    parser.add_argument('-r', '--random', action='store_true', help='run random search in the given domain')
    parser.add_argument('-l', '--lhs',  action='store_true', help='run Latin Hypercube Sampling in the given domain')
    parser.add_argument('-n', '--num_configs', default=100, help='number of hyperparameter configurations to try')

    args = parser.parse_args()

    print(args.param_to_vary)
    param_to_vary = args.param_to_vary[0]
    print(param_to_vary)
    print(args.values)
  
    if len(args.values) != len(args.param_to_vary):
        raise(Exception('Number of hyperparameters does not match number of value ranges'))

    if args.random:
        # random search
        configs_set=[]
        for i in range(int(args.num_configs)):
            config_sample = []
            for param in range(len(args.param_to_vary)):
                config_sample.append(eval(random.sample(args.values[param], 1)[0]))
                # used eval() to turn string to numerical values
                # TODO: enable taking string values also
            configs_set.append(config_sample)
    elif args.lhs:
        pass
        # latin hypercube sampling
    else:
        param_values_list = [args.values[param] for param in range(len(args.param_to_vary))]
        # param_values_list is of the form [ [p1v1, p1v2, ...], [p2v1, p2v2, ...], ... ]
        configs_set = list(itertools.product(*param_values_list))
        # configs_set is the Cartesian product of all param values [ [p1vi, p2vj, p3vk, ...] ]

    print('configs_set {}'.format(configs_set))

    return args, configs_set


# function for running training + evaluating + SAVING output for each configuration (move code from June6 here)
# input configs_set
# note that there are also hyperparameters for the function that evaluates each hyperparameter configuration

def train_and_evaluate(args, config, trial_id, n_iterations = 100, learning_steps = 1000, eval_episodes = 10):

    # initialize environment and RL algorithm
    env = gym.make(str(args.env))
    # to specify the hyperparameters, make a dictionary of the form {'param_to_vary':value} as **kwargs
    #config_values = [eval(config_item) for config_item in config]
    args_dict = dict(zip(args.param_to_vary, config))
    print('args.dict {}'.format(args_dict))
    model = eval(args.algorithm)('MlpPolicy', env, **args_dict)

    """
    n_iterations: number of training-evaluation iterations
    learning_steps: how many steps to run the learning
    eval_episodes: how many episodes to run when evaluating a trained algo
    """

    rewards = np.zeros((n_iterations, 2))
    infos = []

    main_save_dir = args.outputdir+'/'+trial_id
    #os.mkdir(main_save_dir)
    os.makedirs(main_save_dir)
    print("Output directory {} created".format(main_save_dir))
    dill.dump([args, config], open(main_save_dir+'/'+'args_and_configs', 'wb'))

    for n_iter in range(n_iterations):
        print('iteration {} started'.format(n_iter))
        model.learn(total_timesteps = learning_steps)
        print('iteration {} learning completed'.format(n_iter))
        episode_rewards, _, episode_infos, info_names = \
            evaluate_policy(model, env, 
                            n_eval_episodes = eval_episodes, 
                            return_episode_rewards = True)

        reward_mean, reward_std = np.mean(episode_rewards), np.std(episode_rewards)
        episode_infos_array = np.stack(episode_infos)
        episode_infos_mean, episode_infos_std = np.mean(episode_infos_array, axis=0), np.std(episode_infos_array, axis=0)
        
        rewards[n_iter] = np.array([reward_mean, reward_std])
        infos.append([episode_infos_mean, episode_infos_std])
        #wandb.log({'main_reward': reward_mean, 'right_knee': episode_infos_mean[0], 'left knee': episode_infos_mean[1]})
        
        dill.dump(episode_rewards, open(main_save_dir+'/'+str(n_iter)+'_rewards', 'wb'))
        dill.dump(episode_infos, open(main_save_dir+'/'+str(n_iter)+'_infos', 'wb'))

        # TODO: have the directory name / file name reflect the config
        # or save results in csv? more readable / handlable with pandas


    # save output
    # dill.dump(the_object_to_be_saved, save_dir)
    #dill.dump(sim_params, open("{}/sim_params.dill".format(save_dir), "wb"))
    # later, write plotting code that recovers data through dill.load()


# to learn: multiprocessing 

if __name__ == "__main__":
    
    args, configs_set = parse_hp_args()

    env = gym.make(str(args.env))

    trial_id_record = open(args.outputdir+'/trial_id_record_june19_50runs.txt', 'w+')

    for config in configs_set:
        start_time = time.time()

        trial_id = str(time.time()).split('.')[0]
        print("Testing config {}, Using Trial ID: {}".format(dict(zip(args.param_to_vary, config)), trial_id))

        trial_id_record.write(trial_id+', '+str(dict(zip(args.param_to_vary, config)))+'\n')

        train_and_evaluate(args, config, trial_id, n_iterations=100)

        print("Finished running current config iteration, time taken {} s".format(time.time()-start_time))