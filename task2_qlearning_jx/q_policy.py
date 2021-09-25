import numpy as np
import random

rng = np.random.default_rng()


def q_policy(**policy_kwargs):
    agent_blinky = getattr(policy_kwargs['common_observation'], 'agent_blinky')
    ab_distance = abs(agent_blinky[0]) + abs(agent_blinky[1])
    agent_inky = getattr(policy_kwargs['common_observation'], 'agent_inky')
    ai_distance = abs(agent_inky[0]) + abs(agent_inky[1])
    three_square = getattr(policy_kwargs['common_observation'], 'agent_three_square')
    setting = policy_kwargs['setting']
    wall_color = getattr(setting, 'wall_color')
    path_color = getattr(setting, 'path_color')
    ############### please ignore above code #######################

    s = policy_kwargs['s']
    Q = policy_kwargs['Q']
    # # find the optimal Q value action
    actions = policy_kwargs['actions']
    max_value = max([Q(s, a) for a in actions])
    max_actions = [a for a in actions if Q(s, a) == max_value]
    return random.choice(max_actions)

