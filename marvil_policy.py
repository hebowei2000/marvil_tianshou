import torch
import numpy as np
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

        mode: str = 'continuous',
        **kwargs: Any,

    ) -> None:
        super().__init__()
    
    def train(self, mode: bool = True) -> "marvil_policy":
        self.training = mode
        self.policy_net.train(mode)
        self.value_net.train(mode)
        return self

    def forward():


    def learn():
        1
        return {
            "policy_loss": p_loss.item(),
            "vf_loss": v_loss.item(),
            "total_loss": total_loss.item(),
            "vf_explained_var": explained_variance.item()
        }

class marvil_loss:
    def __init__(self, policy, value_estimates,
                 action_dist, actions, cumulative_rewards, vf_loss_coeff, beta):

        #Advantage Estimation
        adv = cumulative_rewards - value_estimates

