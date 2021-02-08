import torch
import numpy as np
import scipy.signal
from copy import deepcopy
from torch.distributions import Independent, Normal
from typing import Any, Dict, Tuple, Union, Optional, Mapping, List

from tianshou.policy import PGPolicy, PPOPolicy, BasePolicy
from tianshou.exploration import BaseNoise
from tianshou.data import Batch, ReplayBuffer, to_torch

"""
   Reimplementation of MARVIL policy with deep reinforcement learning framework tianshou 
"""
class mavil_policy(BasePolicy):
    def __init__(
        self,
        policy_net: torch.nn.Module,
        value_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        discount_factor: float=0.99,
        vf_coeff: float=1.0,
        mode: str = 'continuous',
        **kwargs: Any,

    ) -> None:
        super().__init__()
        self.vf_coeff = vf_coeff


    def train(self, mode: bool = True) -> "marvil_policy":
        self.training = mode
        self.policy_net.train(mode)
        self.value_net.train(mode)
        return self

    def forward(
        self, 
        batch: Batch,
        state: Optional[Union[dict, Batch, np.adarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        obs = batch[input]
        # v_value = self.value_net(obs, state=state, info=batch.info)
        logits, state = self.policy_net(obs, state=state, info=batch.info)
        if self.mode == "discrete":
            action = logits(dim =1)[1]
        else:
            action = logits

        return Batch(act=action, state=state, logits=logits)

    def discount_cumsum(self, x: np.ndarray, gamma: float) -> float:
        """
         Calculate the discounted cumulative sum over a reward sequence 'x'
         
         Args:
            gamma (float): the discount factor gamma

         Returns:
           float: the discounted cumulative sum overe the reward sequence 'x' 
        """
        return scipy.signal.lfilter([1], [1, float(-0.9)], x[::-1], axis=0)[::-1]

    def compuate_advantage(self, batch:Batch, last_r: float, 
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











    def postprocess_advantage(self, policy, batch:Batch, **kwargs: Any):
        #Postprocesses a trajectory and returns the processed trajectory
        if  batch.done[-1]:
            last_r = 0.0

    def learn(self, batch:Batch, **kwargs: Any) -> Dict[str, float]:
        # Advantage estimation

        # Value loss

        # Policy loss
        # Update averaged advantage norm

        # Exponentially weighted advantages

        # log\pi_\theta(a|s)

        # Combine both losses


        self.optim.zero_zero_grad()
        loss.backward()
        self.optim.step()

        return {
            "policy_loss": p_loss.item(),
            "vf_loss": v_loss.item(),
            "total_loss": total_loss.item(),
            "vf_explained_var": explained_variance.item()
        }



