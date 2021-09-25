import numpy as np
import random
from task1_environment.policy.baseline import observe

rng = np.random.default_rng()


class Character:
    def __init__(self, position, policy):
        # set agent initial position
        self.position = self.init_position = position

        # initialise trivial policy
        # self.policy = lambda observation, setting: 0
        self.policy = policy

        # initialise trivial observation_func
        # self.observe = lambda agent_position, environment_array: 0
        self.observe = observe

        self.observation = None

        self.direction_proposal = None

        self.target_proposal = None

        self.last_position = None

    def set_policy(self, policy):
        self.policy = policy

    def set_observe(self, observe_policy):
        self.observe = observe_policy

    def propose_movement(self, **kwargs):
        # If observation is not default,
        # self.observe should be set first
        # i.e., agent.observe = observe_function
        # observe_function should decouple from this class and
        # can be customised

        if 'instruction' in kwargs:
            self.direction_proposal = kwargs['instruction']
        else:
            ob_kwargs = {'self_position': self.position, **kwargs}
            self.observation = self.observe(**ob_kwargs)
            policy_kwargs = {
                'self_position': self.position,
                'observation': self.observation,
                'last_move': self.direction_proposal,
                **kwargs
            }
            self.direction_proposal = self.policy(**policy_kwargs)
        self.target_proposal = self.position + self.direction_proposal

    def reset(self):
        # set agent initial position
        self.position = self.init_position

    def randomise_position(self, possible_cell_color, array, **kwargs):

        # copy maze.array to avoid change it
        array = array.copy()

        # mask agent position
        if 'agent_position' in kwargs:
            array[tuple(kwargs['agent_position'])] = -possible_cell_color

        # mask blinky position
        if 'blinky_position' in kwargs:
            array[tuple(kwargs['blinky_position'])] = -possible_cell_color

        # pool of possible indexes
        cell_candidates_index = \
            np.nonzero((array == possible_cell_color))

        # number of available pairs of indexes
        range_for_random = len(cell_candidates_index[0])
        # pick a random one
        random_index = rng.integers(range_for_random)
        # return the random index
        self.init_position = \
            self.last_position = \
            self.position = \
            np.array((
                cell_candidates_index[0][random_index],
                cell_candidates_index[1][random_index]
            ))


class Agent(Character):
    def __init__(self, name, position, policy):
        super().__init__(position, policy)
        self.name = name


class Ghost(Character):
    def __init__(self, name, position, policy):
        super().__init__(position, policy)
        self.name = name
