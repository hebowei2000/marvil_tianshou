import numpy as np
import tensorflow as tf 
import pdb
import torch
import gym
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv

class Env(HalfCheetahEnv):
    def __init__(self, bnns, env_buffer, config,
                 penalty_coeff=0.,
                 penalty_learned_var=False,
                 penalty_learned_var_random=False):
        self.bnns = bnns
        self.config = config
        self.penalty_coeff = penalty_coeff
        self.penalty_leared_var = penalty_learned_var
        self.penalty_learned_var_random = penalty_learned_var_random
        self.reset()
        super().__init__()

    def reset(self):
        batch, _ = self.env_buffer.sample(1)
        self.obs = batch.obs[0]
        return self.obs

    def _get_logprob(self, x, means, variance):
        k = x.shape[-1]

        ## [num_networks, batch_size] 
        log_prob = -1/2 * (k * np.log(2 * np.pi)) + np.log(variances).sum(-1) + (np.power(x - means, 2).sum(-1))

        ## [batch_size]
        prob = np.exp(log_prob).sum(0)

        ##[batch_size]
        log_prob = np.log(prob + 1e-6)

        stds = np.std(means,0).mean(-1)

        return log_prob, stds

    def step(self, act, deterministic=True):
        obs = self.obs

        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs,act),axis=-1)
        ensemble_model_means, ensemble_model_vars = self.bnns.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(np.exp(ensemble_model_vars))

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds\
        
        if not deterministic:
            ### choose one model from ensemble
            num_models, batch_size, _ = ensemble_model_means.shape
            model_inds = self.bnns.random_inds(batch_size)
            batch_inds = np.arrange(0, batch_size)
            samples = ensemble_samples[model_inds, batch_inds]
            model_means = 