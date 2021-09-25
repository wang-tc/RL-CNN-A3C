import sys
sys.path.append("/Deep-Reinforcement-Learinng-Submission/task1_environment/")

import numpy as np
from task1_environment.environment.main import PacMan
from task1_environment.environment.maze import Maze, Dots
from task1_environment.environment.character import Ghost
from task1_environment.environment.figures import Process
from task1_environment.environment.observation import CommonObservation
from task1_environment.policy.baseline import inky_policy, relative_position_to_direction, safe_delete, observe, rng
from task2_qlearning_jx.q_agent import Agent
from task1_environment.policy.baseline import agent_policy

class PacManJX(PacMan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dots = DotsJX(self.setting, self.maze.array)
        self.agent = Agent('agent', self.setting.agent_init_position, agent_policy)
        self.blinky = Ghost('blinky', self.setting.blinky_init_position, blinky_policy_intelligent)
        # self.inky = Ghost('inky', self.setting.inky_init_position, inky_policy)

        self.setting.reward_dict = {
            self.setting.dot_color: 10,
            self.setting.path_color: -1,
            self.setting.wall_color: -20,
            self.setting.blinky_color: -99,
            self.setting.inky_color: -99
        }

        self.process.win = False
        # False, let ghost(blinky) move randomly / True, let ghost(blinky) move intelligently
        self.blinky_random = True

        if self.blinky_random:
            self.blinky.policy = blinky_policy_random

        self.old_agent = self.agent.position
        self.old_blinky = self.blinky.position
        self.old_inky = self.inky.position
        self.dots_left = round(self.dots.curr_dotsNum / self.dots.init_dotsNum, 2)

    def agent_move(self):
        # agent move
        self.old_agent = self.agent.position
        self.agent.propose_movement(
            synthetic_array=self.synthetic_array,
            setting=self.setting,
            common_observation=self.common_observation,
            s=self.state()
        )
        # compute current reward
        self.compute_current_reward(self.agent.target_proposal, old_agent=self.old_agent,
                                    old_blinky=self.old_blinky, blinky_proposal=self.blinky.position,
                                    old_inky=self.old_inky, inky_proposal=self.inky.position)
        self.process.reward += self.process.current_reward
        # print(self.process.current_reward)
        # check agent move validation
        self.maze.validate_proposal(self.agent.target_proposal)
        if self.maze.agent_movement_validation:
            self.agent.position = self.agent.target_proposal
        self.dots_left = round(self.dots.curr_dotsNum / self.dots.init_dotsNum, 2)

    def ghost_move(self):
        self.old_blinky = self.blinky.position
        self.old_inky = self.inky.position
        for ghost in [self.blinky, self.inky]:
            ghost.propose_movement(
                synthetic_array=self.synthetic_array,
                setting=self.setting,
                common_observation=self.common_observation,
                blinky_random=self.blinky_random
            )
            ghost.position = ghost.target_proposal

    def state(self):
        """Return a description of the state of the environment."""
        s = {'neighbours': self.common_observation.agent_three_square, 'agent': tuple(self.agent.position),
             'blinky': tuple(self.blinky.position), 'inky': tuple(self.inky.position)}
        s = []
        neighbours = self.common_observation.agent_three_square.flatten()
        agent = np.array(self.agent.position)
        blinky = np.array(self.blinky.position)
        inky = np.array(self.inky.position)
        dots = np.array(self.dots.curr_dotsNum)
        temp = np.append(inky,dots)
        states = np.concatenate((neighbours,agent,blinky,inky))
        return states

    def random_reset(self):
        # randomise agent position
        self.agent.randomise_position(
            self.setting.path_color,
            self.maze.array
        )
        # randomise ghost blinky positions
        self.blinky.randomise_position(
            self.setting.path_color,
            self.maze.array,
            agent_position=self.agent.position
        )
        # randomise ghost inky positions
        self.inky.randomise_position(
            self.setting.path_color,
            self.maze.array,
            agent_position=self.agent.position,
            blinky_position=self.blinky.position
        )

        # reset dots
        self.dots = DotsJX(self.setting, self.maze.array)
        self.dots_left = self.dots.curr_dotsNum / self.dots.init_dotsNum
        # reset frame
        self.update_synthetic_graph_array()

        self.graphic.frame.set_data(self.synthetic_array)

        # reset process
        self.process = Process()
        self.common_observation = CommonObservation(
            self.synthetic_array,
            agent_position=self.agent.position,
            blinky_position=self.blinky.position,
            inky_position=self.inky.position
        )
        self.animation = None

    def check_termination(self):
        if np.sum(self.dots.array) == 0:
            self.process.termination = True
            self.process.win = True
        if np.all(self.agent.position == self.blinky.position) or \
           np.all(self.agent.position == self.inky.position) or \
           (tuple(self.old_blinky), tuple(self.old_agent)) == (
           tuple(self.agent.position), tuple(self.blinky.position)) or \
           (tuple(self.old_inky), tuple(self.old_agent)) == (
           tuple(self.agent.position), tuple(self.inky.position)):
            self.process.current_reward = self.setting.reward_dict[self.setting.blinky_color]
            self.process.reward += self.process.current_reward
            self.process.termination = True
        if self.process.time == self.setting.maximum_time:
            self.process.termination = True

    def compute_current_reward(self, agent_proposal, **kwargs):
        target_position_color = self.synthetic_array[tuple(agent_proposal)]
        old_agent = kwargs['old_agent']
        blinky_proposal = kwargs['blinky_proposal']
        old_blinky = kwargs['old_blinky']
        inky_proposal = kwargs['inky_proposal']
        old_inky = kwargs['old_inky']

        # if (tuple(old_blinky), tuple(old_agent)) == (
        #    tuple(agent_proposal), tuple(blinky_proposal)) or \
        #    (tuple(old_inky), tuple(old_agent)) == (
        #    tuple(agent_proposal), tuple(inky_proposal)):
        #     self.process.current_reward = self.setting.reward_dict[self.setting.blinky_color]
        # else:
        # self.process.current_reward = self.setting.reward_dict[(target_position_color)]

        target_position_color = self.maze.array[tuple(agent_proposal)]
        target_position_dot = self.dots.array[tuple(agent_proposal)]
        if target_position_dot == self.setting.dot_color:
            self.process.current_reward = self.setting.reward_dict[target_position_dot]
        # elif (tuple(self.old_blinky), tuple(self.old_agent)) == (tuple(self.agent.position), tuple(self.blinky.position)) or \
        #         (tuple(self.old_inky), tuple(self.old_agent)) == (tuple(self.agent.position), tuple(self.inky.position)):
        #
        else:
            self.process.current_reward = self.setting.reward_dict[target_position_color]



def blinky_policy_intelligent(**policy_kwargs):
    candidates = {}
    for k, v in zip(['right', 'down', 'up', 'left'], [(0, 1), (1, 0), (-1, 0), (0, -1)]):
        candidates[k] = np.array(v)
    to_be_delete_list = []
    to_be_delete_set = set()
    agent_blinky = getattr(policy_kwargs['common_observation'], 'agent_blinky')
    three_square = getattr(policy_kwargs['common_observation'], 'blinky_three_square')
    setting = policy_kwargs['setting']
    wall_color = getattr(setting, 'wall_color')
    blinky_random = policy_kwargs['blinky_random']

    blinky_agent_direction = relative_position_to_direction(agent_blinky)

    # delete direction which will lead to wall
    for candidate in candidates:
        target = np.array((1, 1)) + candidates[candidate]
        if three_square[tuple(target)] == wall_color and candidate not in to_be_delete_set:
            to_be_delete_list.append(candidate)
            to_be_delete_set.add(candidate)

    # delete direction which will not lead to agent
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



class DotsJX(Dots):
    def __init__(self, setting, array):
        super().__init__(setting, array)
        self.init_dotsNum = np.count_nonzero(self.array == self.color)
        self.curr_dotsNum = np.count_nonzero(self.array == self.color)

    def mask(self):
        self.curr_dotsNum = np.count_nonzero(self.array == self.color)
        return self.array == self.color