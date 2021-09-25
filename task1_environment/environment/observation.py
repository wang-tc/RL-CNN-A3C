import numpy as np


def three_square_observe(my_position, array):
    """Calculate agent's 3*3 neighbours matrix"""
    array = array.copy()
    surroundings = array[
                   my_position[0] - 1: my_position[0] + 2,
                   my_position[1] - 1: my_position[1] + 2
                   ]
    return surroundings


# default state function, return agent surrouding cells and absolute positions of all characters.
def calculate_state(**kwargs):
    self_position = kwargs['self_position']
    three_square = getattr(kwargs['common_observation'], 'agent_three_square')
    blinky = np.array(kwargs['blinky_p'])
    inky = np.array(kwargs['inky_p'])
    state = tuple(np.concatenate((three_square.flatten(), self_position, blinky, inky)))
    return state


# This class is record common observations, to be used conveniently in other tasks.
# three squares means surrounding cells and agent_blinky means relative position from agent to blinky and so on.
class CommonObservation:
    def __init__(self, maze_array, **kwargs):
        self.agent_three_square = None
        self.blinky_three_square = None
        self.inky_three_square = None
        self.agent_blinky = None
        self.agent_inky = None
        self.update(maze_array, **kwargs)

    def set_relative_positions(self, **kwargs):
        self.agent_blinky = kwargs['blinky_position'] - kwargs['agent_position']
        self.agent_inky = kwargs['inky_position'] - kwargs['agent_position']

    def set_three_square(self, maze_array, **kwargs):
        maze_array = maze_array.copy()
        for character in ['agent', 'blinky', 'inky']:
            key = character + '_position'
            # observe 3**2 maze based on character's position which is kwargs[key]
            square = three_square_observe(kwargs[key], maze_array)
            setattr(self, character + '_three_square', square)

    def update(self, maze_array, **kwargs):
        self.set_relative_positions(**kwargs)
        self.set_three_square(maze_array, **kwargs)
