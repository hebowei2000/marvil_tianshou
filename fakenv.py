import numpy as np
import tensorflow as tf 
import pdb
import torch
import gym
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv

class FakeEnv(HalfCheetahEnv):
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
            model_means = ensemble_model_means[model_inds, batch_inds]
            model_stds = ensemble_model_stds[model_inds, batch_inds]

        else:
            samples = np.mean(ensemble_samples, axis=0)
            model_means = np.mean(ensemble_model_means, axis=0)
            model_stds = np.mean(ensemble_model_stds, axis=0)

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)
        
        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.config.termination_fn(obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:,:1], terminals, model_means[:,1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:,:1], terminals, model_stds[:,1:]), axis=-1)

        if self.penalty_coeff !=0:
            if not self.penalty_learned_var:
                ensemble_means_obs = ensemble_model_means[:,:,1:]
                mean_obs_means = np.mean(ensemble_means_obs, axis=0)
                diffs = ensemble_means_obs - mean_obs_means
                normalize_diffs = False
                if normalize_diffs:
                    obs_dim = next_obs.shape[1]
                    obs_sigma = self.bnns.scaler.cached_sigma[0,:obs_dim]
                    diffs = diffs / obs_sigma
                dists = np.linalg.norm(diffs, axis=2)
                penalty = np.max(dists, axis=0)
            else:
                penalty = np.amax(np.linalg.norm(ensemble_model_stds, axis=2), axis=0)

            penalty = np.expand_dims(penalty, 1)
            assert penalty.shape == rewards.shape
            unpenalized_rewards = rewards
            penalized_rewards = rewards - self.penalty_coeff * penalty
        else:
            penalty = None
            unpenalized_rewards = rewards
            penalized_rewards = rewards

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            unpenalized_rewards = unpenalized_rewards[0]
            penalized_rewards = penalized_rewards[0]
            terminals = terminals[0]
            
        self.obs = next_obs
        info = {'mean': return_means, 'std': return_stds, 'log_prob': logh_prob, 'dev': dev,
                'unpenalized_rewards': unpenalized_rewards, 'penalty': penalty, 'penalized_rewards': penalized_rewards}
        
        return next_obs, penalized_rewards, terminals, info
        
    def close(self):
        pass
