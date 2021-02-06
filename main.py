import os
import gym
import torch
import pickle
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offline_trainer
from marvil_policy import marvil_policy
from bnn import EnsembledBNN
from fakenv import FakeEnv

class StaticFns:
    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        done = np.array([False]).repeate(len(obs))
        done = done[:,None]
        return done

    @staticmethod
    def recompute_reward_fn(obs, act, next_obs, rew):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        new_rew = -(rew + 0.1 * np.sum(np.square)(act))) - 0.1 * np.sum(np.square(act))
        return new_rew

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="halfcheetah-medium-v0")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--buffer-size', type=int, default=200000)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--logdir', type=str, default='log')
    return parser.parse_args()

def main(args=get_args()):
    envs = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print('Running on ' + args.device)
    dataset = d4rl.qlearning_dataset(env)

    ######################################### Buffer ############################
    # Buffer for offline data
    # load data, most d4rl dataset has 1M SARSA pair
    obs_array=dataset['observations']
    act_array=dataset['actions']
    rew_array=dataset['rewards']
    done_array=dataset['terminals']
    next_obs_array=dataset['next_observations']
    offline_buffer=Replaybuffer(size=act_array.shape[0])
    for i in range(act_array.shape[0]):
        offline_buffer.add(obs=obs_array[i], act=act_array[i], rew=rew_array[i], done=done_array[i], obs_next=next_obs_array[i], info={})

    # Buffer for policy learning
    marvil_buffer = Replaybuffer(args.buffer_size)

    ######################################### Policy ############################

    ######################################### Create Environment ################
    

    ######################################### Collector #########################
    # For offline setting, train_collector is no longer needed
    test_collector = Collector(policy, test_envs)

    ######################################### Log ###############################
    log_path = os.path.join(args.logdir, args.task, 'marvil')
    writer = SummaryWriter(log_path)
    
    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold

    ######################################### Train #############################
    # trainer


if __name__ == '__main__':
    main()

