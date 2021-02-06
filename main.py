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
from tianshou.trainer import offline_trainer, off
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
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--render', type=float, default=0.)

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
    # Dummy env or single env are both accepted because the collector will convert single env 
    # to dummy env later.
    # For marvil, train_env is no longer needed and test_env should be the real env.
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)

    # Actor Network


    # Value Network






    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path))
        print("Loaded agent from: ", args.resume_path)
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
    result = offline_trainer(
        policy, offline_buffer, test_collector, args.epoch,
        args.step_per_epoch, args.test_num, args.batch_size,
        stop_fn=stop_fn, save_fn=save_fn, writer=writer
    )

    assert stop_fn(result['best_reward'])

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        policy.eval()
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=[1] * args.test_num, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        
if __name__ == '__main__':
    main()

