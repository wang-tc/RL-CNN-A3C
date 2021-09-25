# https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html#naive-q-learning

import random
from task1_environment.policy.constants import ACTION_SPACE


# Agent policy for Q learning
def q_policy(state, table, **kwargs):
    qvals = {action: table[state, action] for action in ACTION_SPACE}
    max_q_val = max(qvals.values())
    actions_with_max_q_val = [action for action, qval in qvals.items() if qval == max_q_val]
    return random.choice(actions_with_max_q_val)
