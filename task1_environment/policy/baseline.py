import numpy as np

rng = np.random.default_rng()


def relative_position_to_direction(array):
    """
    convert relative postion to direction such as:
     (2,3) --> (0, 1)
     (-5, 3) --> (-1, 0)
    :param array: np.ndarray, wi
    :return: np.ndarray
    """
    ans = []
    array = array.copy()
    array2 = array.copy()
    if array[0] == 0 and array[1] == 0:
        return [array, array2]
    elif array[1] == 0:
        array[0] /= abs(array[0])
        array2 = array
    elif array[0] == 0:
        array2[1] /= abs(array2[1])
        array = array2
    else:
        array[0] /= abs(array[0])
        array[1] = 0
        array2[0] = 0
        array2[1] /= abs(array2[1])
    ans.extend([array, array2])

    return ans


def observe(**ob_kwargs):
    return ob_kwargs


def safe_delete(dictionary, key):
    if len(dictionary) > 1:
        del dictionary[key]
    return dictionary

# intelligent agent policy
def agent_policy(**policy_kwargs):
    candidates = {}
    for k, v in zip(['right', 'down', 'up', 'left'], [(0, 1), (1, 0), (-1, 0), (0, -1)]):
        candidates[k] = np.array(v)
    agent_blinky = getattr(policy_kwargs['common_observation'], 'agent_blinky')
    ab_distance = abs(agent_blinky[0]) + abs(agent_blinky[1])
    agent_inky = getattr(policy_kwargs['common_observation'], 'agent_inky')
    ai_distance = abs(agent_inky[0]) + abs(agent_inky[1])
    three_square = getattr(policy_kwargs['common_observation'], 'agent_three_square')
    setting = policy_kwargs['setting']
    wall_color = getattr(setting, 'wall_color')
    path_color = getattr(setting, 'path_color')

    to_be_delete_set = set()
    to_be_delete_list = []

    agent_blinky_direction = relative_position_to_direction(agent_blinky)
    agent_inky_direction = relative_position_to_direction(agent_inky)

    # delete direction which will lead to wall
    for candidate in candidates:
        target = np.array((1, 1)) + candidates[candidate]
        if three_square[tuple(target)] == wall_color and candidate not in to_be_delete_set:
            to_be_delete_list.append(candidate)
            to_be_delete_set.add(candidate)

    # deletes options that leads to blinky
    if ab_distance <= 3:
        for candidate in candidates:
            direction = candidates[candidate]
            if ((direction == agent_blinky_direction[0]).all() or
                (direction == agent_blinky_direction[-1]).all()) and \
                    candidate not in to_be_delete_set:
                to_be_delete_list.append(candidate)
                to_be_delete_set.add(candidate)

    # deletes options that leads to inky
    if ai_distance <= 3:
        for candidate in candidates:
            direction = candidates[candidate]
            if ((direction == agent_inky_direction[0]).all() or
                (direction == agent_inky_direction[-1]).all()) and \
                    candidate not in to_be_delete_set:
                to_be_delete_list.append(candidate)
                to_be_delete_set.add(candidate)

    # delete going to path options
    for candidate in candidates:
        target = np.array((1, 1)) + candidates[candidate]
        if three_square[tuple(target)] == path_color and candidate not in to_be_delete_set:
            to_be_delete_list.append(candidate)
            to_be_delete_set.add(candidate)

    for key in to_be_delete_list:
        safe_delete(candidates, key)

    return list(v for k, v in candidates.items())[0]

# intelligent blinky policy
def blinky_policy(**policy_kwargs):
    candidates = {}
    for k, v in zip(['right', 'down', 'up', 'left'], [(0, 1), (1, 0), (-1, 0), (0, -1)]):
        candidates[k] = np.array(v)

    to_be_delete_list = []
    to_be_delete_set = set()
    agent_blinky = getattr(policy_kwargs['common_observation'], 'agent_blinky')
    three_square = getattr(policy_kwargs['common_observation'], 'blinky_three_square')
    setting = policy_kwargs['setting']
    last_move = policy_kwargs['last_move']
    wall_color = getattr(setting, 'wall_color')
    blinky_agent_direction = relative_position_to_direction(agent_blinky)

    # eliminate direction which will lead to wall
    for candidate in candidates:
        target = np.array((1, 1)) + candidates[candidate]
        if three_square[tuple(target)] == wall_color and candidate not in to_be_delete_set:
            to_be_delete_list.append(candidate)
            to_be_delete_set.add(candidate)

    # eliminate options that is the reverse of last movement, so the
    # ghost will not be trapped in two cells
    if last_move is not None:
        for candidate in candidates:
            direction = candidates[candidate]
            if (direction == -last_move).all() and candidate not in to_be_delete_set:
                to_be_delete_list.append(candidate)
                to_be_delete_set.add(candidate)

    # delete direction which will not lead to agent
    # print(blinky_agent_direction, -blinky_agent_direction[0], -blinky_agent_direction[-1])
    for candidate in candidates:
        direction = candidates[candidate]
        if not ((direction == -blinky_agent_direction[0]).all()) and \
                not ((direction == -blinky_agent_direction[-1]).all()) and \
                candidate not in to_be_delete_set:
            to_be_delete_list.append(candidate)
            to_be_delete_set.add(candidate)

    for key in to_be_delete_list:
        safe_delete(candidates, key)

    return list(v for k, v in candidates.items())[0]

# general random policy
def random_policy(**policy_kwargs):
    name = policy_kwargs['name']
    candidates = {}
    for k, v in zip(['right', 'down', 'up', 'left'], [(0, 1), (1, 0), (-1, 0), (0, -1)]):
        candidates[k] = np.array(v)
    to_be_delete_list = []
    to_be_delete_set = set()
    setting = policy_kwargs['setting']
    wall_color = getattr(setting, 'wall_color')
    three_square = getattr(policy_kwargs['common_observation'], name + '_three_square')
    for candidate in candidates:
        target = np.array((1, 1)) + candidates[candidate]
        if three_square[tuple(target)] == wall_color and candidate not in to_be_delete_set:
            to_be_delete_list.append(candidate)
            to_be_delete_set.add(candidate)
    for key in to_be_delete_list:
        safe_delete(candidates, key)
    return rng.choice(list(v for k, v in candidates.items()))

# blinky random policy
def blinky_policy_random(**policy_kwargs):
    candidates = {}
    for k, v in zip(['right', 'down', 'up', 'left'], [(0, 1), (1, 0), (-1, 0), (0, -1)]):
        candidates[k] = np.array(v)
    to_be_delete_list = []
    to_be_delete_set = set()
    agent_blinky = getattr(policy_kwargs['common_observation'], 'agent_blinky')
    three_square = getattr(policy_kwargs['common_observation'], 'blinky_three_square')
    setting = policy_kwargs['setting']
    wall_color = getattr(setting, 'wall_color')

    blinky_agent_direction = relative_position_to_direction(agent_blinky)

    # delete direction which will lead to wall
    for candidate in candidates:
        target = np.array((1, 1)) + candidates[candidate]
        if three_square[tuple(target)] == wall_color and candidate not in to_be_delete_set:
            to_be_delete_list.append(candidate)
            to_be_delete_set.add(candidate)
    for key in to_be_delete_list:
        safe_delete(candidates, key)
    return rng.choice(list(v for k, v in candidates.items()))

# inky random policy
def inky_policy(**policy_kwargs):
    candidates = {}
    for k, v in zip(['right', 'down', 'up', 'left'], [(0, 1), (1, 0), (-1, 0), (0, -1)]):
        candidates[k] = np.array(v)
    to_be_delete_list = []
    to_be_delete_set = set()
    setting = policy_kwargs['setting']
    wall_color = getattr(setting, 'wall_color')
    three_square = getattr(policy_kwargs['common_observation'], 'inky_three_square')
    for candidate in candidates:
        target = np.array((1, 1)) + candidates[candidate]
        if three_square[tuple(target)] == wall_color and candidate not in to_be_delete_set:
            to_be_delete_list.append(candidate)
            to_be_delete_set.add(candidate)
    for key in to_be_delete_list:
        safe_delete(candidates, key)
    return rng.choice(list(v for k, v in candidates.items()))
