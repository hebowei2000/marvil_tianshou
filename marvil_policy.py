import torch
import numpy as np
from copy import deepcopy
from torch.distributions import Independent, Normal
from typing import Any, Dict, Tuple, Union, Optional, Mapping, List

from tianshou.policy import PGPolicy, PPOPolicy, BasePolicy
from tianshou.exploration import BaseNoise
from tianshou.data import Batch, ReplayBuffer, to_torch

class mavil_policy(BasePolicy):
    11

class marvil_loss:
    def __init__(self, policy, value_estimates,
                 action_dist, actions, cumulative_rewards, vf_loss_coeff, beta):

        #Advantage Estimation
        adv = cumulative_rewards - value_estimates

