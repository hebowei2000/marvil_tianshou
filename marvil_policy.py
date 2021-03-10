import torch
import numpy as np
import scipy.signal
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from torch.distributions import Independent, Normal
from typing import Any, Dict, Tuple, Union, Optional, Mapping, List, Callable, Type
from tianshou.policy import PGPolicy, PPOPolicy, BasePolicy
from tianshou.exploration import BaseNoise
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy

"""
   Reimplementation of MARVIL policy with deep reinforcement learning framework tianshou 
"""
class mavil_policy(BasePolicy):
    def __init__(
        self,
        policy_net: torch.nn.Module,
        value_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dist_fn: Callable[[], torch.distributions.Distribution],
        discount_factor: float=0.99,
        vf_coeff: float=1.0,
        mode: str = 'continuous',
        reward_normalization: bool = False,
        **kwargs: Any,

    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model: torch.nn.Module = model
        self.vf_coeff = vf_coeff
        assert 0.0 <= discount_factor <= 1.0
        self.gamma = discount_factor
        self.dist_fn = dist_fn
        self.rew_normalization = reward_normalization


    def train(self, mode: bool = True) -> "marvil_policy":
        self.training = mode
        self.policy_net.train(mode)
        self.value_net.train(mode)
        return self

    def forward(
        self, 
        batch: Batch,
        state: Optional[Union[dict, Batch, np.adarray]] = None,
        **kwargs: Any,
    ) -> Batch:
     
     """
        Compute action over the given batch data
        stochastic action distribution 
        Return: A:class: 'tianshou.data.Batch' which has 4 keys:
              * "act" the action
              * "logits" the network's raw output
              * "state" the hidden state
     """
        
        # v_value = self.value_net(obs, state=stategamma, info=batch.info)
        logits, h = self.policy_net(batch.obs, state=state, info=batch.info)
        #if self.mode == "discrete":
        #    action = logits(dim =1)[1]
        #else:
        #    action = logits
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)

        act = dist.sample()
        return Batch(logits=logits, act=act, state=h, dist=dist)

    def discount_cumsum(self, x: np.ndarray, gamma: float) -> float:
        """
         Calculate the discounted cumulative sum over a reward sequence 'x'
         
         Args:
            gamma (float): the discount factor gamma

         Returns:
           float: the discounted cumulative sum overe the reward sequence 'x' 
        """
        return scipy.signal.lfilter([1], [1, float(-0.9)], x[::-1], axis=0)[::-1]

    def compute_advantage(self, batch:Batch, last_r: float, 
                           gamma: float = 0.9, lamda: float = 1.0, use_gae: bool = True, use_critic: bool = True):
        """
         Given a rollout, compute its value targets and the advantage
         Args: batch (Batch): batch of a single trajectory
               last_r (float): value estimation for the last observation
               gamma (float): Discount factor
               lambda (float): parameter for GAE
               use_gae (bool): using Generalized Advantage Estimation
               use_critic (bool): whether to use critic (value estimation), setting this to false will use 0 as baseline
        
         Returns: batch (Batch): object with experience from batch and processed rewards
        """  

        assert batch.vf_preds in batch or not use_critic
        assert use_critic or not use_gae

        if use_gae:
            vpred_t = np.concatenate([batch.vf_preds, np.array([last_r])])
            delta_t = (batch.rew + gamma * vpred_t[1:] - vpred_t[:-1])
            # This formula for the advantage comes from "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
            batch.advantages = self.discount_cumsum(delta_t, gamma * lamda)
            batch.value_targets = (batch.advantages + batch.vf_preds).astype(np.float32)

        else:
            rewards_plus_v = np.concatenate([batch.rew, np.array([last_r])])
            discounted_returns = discount_cumsum(rewards_plus_v, gamma)[:-1].astype(np.float32)

            if use_critic:
                batch.advantages = discounted_returns - batch.vf_preds
                batch.value_targets = discounted_return
            else:
                batch.advantages = discounted_returns
                batch.value_targets = np.zeros_like(batch.advantages)

        batch.advantages = batch.advantages.astype(np.float32)

        return batch















    def explained_variance(y, pred):
        y_var = torch.var(y, dim=[0])
        diff_var = torch.var(y - pred, dim=[0])
        min_ = torch.tensor([-1.0]).to(pred.device)
        return torch.max(min_, 1 - (diff_var / y_var)) 

    def postprocess_advantage(self, policy, batch:Batch, **kwargs: Any):
        #Postprocesses a trajectory and returns the processed trajectory
        # Trajectory is actually complete -> last r = 0.0
        if  batch.done[-1]:
            last_r = 0.0
        # Trajectory has been truncated -> last r=VF estimate of last obs
        else:
             # Input dict is provided to us automatically via the Model's requirements.
             # It's a single-timestep (last one in the trajectory) input dict.
                                                                                                                   

        return compute_advantage(
            batch,
            last_r,
            self.gamma,
            # We just want the discounted cummulative rewards, so we won't need
            #  GAE nort critic(use_critic=True: Subtract vf-estimates from returns).
            use_gae = False,
            use_critic = False )

    def learn(self, batch:Batch, batch_size: int, repeat: int, **kwargs: Any) -> Dict[str, float]:
        
        for _ in range(repeat):
            for b in batch.split(batch_size, merge_last=True):
                self.optim.zero_grad()
                state_values = self.value_net(b.obs, b.act).flatten()
                advantages = b.advantages.flatten()
                #actions = b.act
                dist = self(b).dist
                actions = to_torch_as(b.act, dist.logits)
                rewards = to_torch_as(b.returns, dist.logits)
                # Advantage estimation
                adv = advantages - state_values
                adv_squared = (adv.pow(2)).mean()

                # Value loss
                v_loss = 0.5 * adv_squared

                # Policy loss
                # Update averaged advantage norm

                # Exponentially weighted advantages
                exp_advs = 
                # log\pi_\theta(a|s)
                log_prob = dist.log_prob(a).reshape(len(rewards), -1).transpose(0, 1)
                p_loss = - 1.0 * (log_prob * exp_advs.detach()).mean()

                # Combine both losses
                loss = p_loss + self.vf_coeff * v_loss

                
                loss.backward()
                self.optim.step()

        return {
            "policy_loss": p_loss.item(),
            "vf_loss": v_loss.item(),
            "total_loss": loss.item(),
            "vf_explained_var": explained_variance.item()
        }



