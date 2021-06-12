import os, sys
import inspect
sys.path.append('~/.local/lib/python3.7/site-packages')
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet_envs
import time
import wandb
from stable_baselines3 import A2C, SAC, PPO, TD3, HER, DQN
from Evaluate_training_performance import evaluate_policy

wandb.init()

env = gym.make('HumanoidBulletEnv-ST-v0')
model = A2C('MlpPolicy', env, verbose = 1)

n_iterations = 100

learning_steps = 1000
eval_episodes = 10
sim_steps_upper_bound = 1000

rewards = np.zeros((n_iterations, 2))
infos = []

for n_iter in range(n_iterations):
    print('iteration {} started'.format(n_iter))
    model.learn(total_timesteps = learning_steps)
    print('iteration {} learning completed'.format(n_iter))
    episode_rewards, episode_lengths, episode_infos, info_names = \
        evaluate_policy(model, env, 
                        n_eval_episodes = eval_episodes, 
                        sim_steps_upper_bound = sim_steps_upper_bound ,
                        return_episode_rewards = True)

    reward_mean, reward_std = np.mean(episode_rewards), np.std(episode_rewards)
    episode_infos_array = np.stack(episode_infos)
    episode_infos_mean, episode_infos_std = np.mean(episode_infos_array, axis=0), np.std(episode_infos_array, axis=0)
    
    rewards[n_iter] = np.array([reward_mean, reward_std])
    infos.append([episode_infos_mean, episode_infos_std])
    wandb.log({'main_reward': reward_mean, 'right_knee': episode_infos_mean[0], 'left knee': episode_infos_mean[1]})


#env.render(mode='human')

"""If I try to render, I get error:
X11 functions dynamically loaded using dlopen/dlsym OK!
cannot connect to X server
"""


# 28-dim state vector 

# next: connect this to stable-baselines
# write code framework that allows for hyperparam tuning, storing trace info
# with the goal of producing training samples for supervised learning


'''
for i_episode in range(1):
    observation = env.reset()
    for t in range(10):
        #env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation)
        print(info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
'''