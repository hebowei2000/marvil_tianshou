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
    parser.add_argument("--task", type=str, default="CartPole-v0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main(args=get_args()):
    envs = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    