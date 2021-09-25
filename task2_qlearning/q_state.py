import math
import collections
import numpy as np


def find_nearest_dot(array, agent_position, setting):
    agent_position = tuple(agent_position)
    que = collections.deque([(agent_position, 0)])
    visited = set()
    while que:
        current_position, distance = que.popleft()
        visited.add(current_position)
        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_position = current_position[0] + direction[0], current_position[1] + direction[1]
            if array[next_position] == setting.dot_color:
                return next_position, distance
            if next_position in visited or array[next_position] == setting.wall_color:
                continue
            else:
                que.append((next_position, distance + 1))

    return -math.inf, -math.inf

# reduce relative position, to ignore any values that abs(value)>2 to reduce the number of total states
def reduce(position):
    x, y = min(position[0], 2), min(position[1], 2)
    x, y = max(x, -2), max(y, -2)
    return x, y

# explained in the report
def simple_state(game):
    three_square = game.common_observation.agent_three_square
    agent_position = game.agent.position
    blinky_position = game.blinky.position
    inky_position = game.inky.position
    state = tuple(np.concatenate((three_square.flatten(), agent_position, blinky_position, inky_position)))
    return state

# explained in the report
def relative_state(game):
    three_square = game.common_observation.agent_three_square
    agent_position = game.agent.position
    setting = game.setting
    synthetic_array = game.synthetic_array.copy()
    # the relative position between agent and blinky
    agent_blinky = game.common_observation.agent_blinky.copy()
    agent_blinky = reduce(agent_blinky)
    # the relative position between agent and inky
    agent_inky = game.common_observation.agent_inky.copy()
    agent_inky = reduce(agent_inky)

    # the relative position of the nearest dot
    nearest_dot, dot_distance = find_nearest_dot(synthetic_array, agent_position, setting)
    if dot_distance == math.inf or dot_distance == -math.inf:
        dot_distance = min(abs(agent_blinky[0])+abs(agent_blinky[1]), abs(agent_inky[0])+abs(agent_inky[1]))
    agent_dot = np.array(nearest_dot) - np.array(agent_position)
    agent_dot_p = np.array(reduce(agent_dot))
    # if dot_distance == math.inf or dot_distance == -math.inf:
        # print(abs(agent_blinky[0])+abs(agent_blinky[1]),abs(agent_inky[0])+abs(agent_inky[1]))
    dot_dis = np.array(dot_distance).reshape(1)
    # if dot_distance == math.inf or dot_distance == -math.inf:
    #     print(dot_dis)

    # check the 'up', 'down', 'left', 'right' positions have wall
    d = [1 if three_square[p] == setting.wall_color else 0 for p in [(0, 1), (2, 1), (1, 0), (1, 2)]]

    # states definition
    state = np.concatenate([agent_blinky, agent_inky, np.array(d), dot_dis, np.array(agent_dot_p)])
    state = tuple(state)
    # print(state)
    return state
