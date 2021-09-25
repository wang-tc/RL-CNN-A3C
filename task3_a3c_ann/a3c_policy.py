import numpy as np
import random

from task1_environment.environment.character import Character
from task1_environment.policy.baseline import observe
from task3_a3c_ann.utils import v_wrap, set_init, push_and_pull
rng = np.random.default_rng()


class A3C_Policy:
    def __init__(self,action):
        self.action = action
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    def policy(self,**policy_kwargs):
        return self.actions[self.action]

from task3_a3c_ann.utils import v_wrap, set_init, push_and_pull

def A3C_Policy_test(**policy_kwargs):
    s = policy_kwargs['s']
    Q = policy_kwargs['Q']
    net = policy_kwargs['net']
    # # find the optimal Q value action
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    action = net.choose_action(v_wrap(s[None, :]))
    return actions[action]

def A3C_Policy_test2(**policy_kwargs):
    s = policy_kwargs['s']
    Q = policy_kwargs['Q']
    net = policy_kwargs['net']
    # # find the optimal Q value action
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    action = net.choose_action(s)
    return actions[action]