# pacman main framework

# Standard library imports
# import typing
import sys

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from task1_environment.policy.baseline import agent_policy, blinky_policy, random_policy, blinky_policy_random, inky_policy
from matplotlib.animation import FuncAnimation

# Local application imports
# if using relative importing, i.e., "from .maze..." it will raise error because
# the value of __name__ is __main__ and there is no such file as __main__.maze
from task1_environment.environment.maze import Maze, Dots
from task1_environment.environment.character import Agent, Ghost
from task1_environment.environment.figures import Process
from task1_environment.environment.figures import Setting
from task1_environment.environment.graphic import Graphic
from task1_environment.environment.observation import CommonObservation, calculate_state

# set numpy array output width
np.set_printoptions(linewidth=100)


class PacMan:
    def __init__(self, **kwargs):
        # order is important

        # setting
        self.setting = Setting(**kwargs)

        # initiate maze
        self.maze = Maze(self.setting)

        # initiate agent
        # default agent position, if want to randomise this,
        # please use game=Pacman(), game.random_reset()
        self.agent = Agent('agent', self.setting.agent_init_position, agent_policy)

        # ghost blinky
        # default blinky position
        self.blinky = Ghost('blinky', self.setting.blinky_init_position, blinky_policy_random)
        self.inky = Ghost('inky', self.setting.inky_init_position, inky_policy)

        # dots
        self.dots = Dots(self.setting, self.maze.array)

        # synthetic graph
        # set synthetic_array as None first because the corresponding method does not return and it's
        # better to set all instance variable in __init__ method
        self.synthetic_array = None
        self.update_synthetic_graph_array()

        # process
        self.process = Process()
        self.show_rewards = False

        # common_observation
        self.common_observation = CommonObservation(
            self.synthetic_array,
            agent_position=self.agent.position,
            blinky_position=self.blinky.position,
            inky_position=self.inky.position
        )

        # initiate graphic
        self.graphic = Graphic(self.synthetic_array)

        # initiate animation
        self.animation = None

        # jrx ************************************************************
        # self.setting.reward_dict = {
        #     self.setting.dot_color: 10,
        #     self.setting.path_color: -1,
        #     self.setting.wall_color: -20,
        #     self.setting.blinky_color: -99
        # }
        # ***************************************************************

    # reward function
    def compute_current_reward(self, proposal):

        # target_position_color = self.synthetic_array[tuple(proposal)]
        # self.process.current_reward = self.setting.reward_dict[int(target_position_color)]
        #
        # # print(self.process.current_reward)
        # if self.show_rewards:
        #     print('\n', 'rewards:', self.process.current_reward)
        # if np.sum(self.dots.array) == 0:
        #     # self.process.current_reward += 1000
        #     self.process.win = True

        # jrx # ***************************************************************
        target_position_color = self.maze.array[tuple(proposal)]
        target_position_dot = self.dots.array[tuple(proposal)]
        if target_position_dot == self.setting.dot_color:
            self.process.current_reward = self.setting.reward_dict[target_position_dot]
        # elif (tuple(self.old_blinky), tuple(self.old_agent)) == (tuple(self.agent.position), tuple(self.blinky.position)) or \
        #         (tuple(self.old_inky), tuple(self.old_agent)) == (tuple(self.agent.position), tuple(self.inky.position)):
        #
        else:
            self.process.current_reward = self.setting.reward_dict[target_position_color]
        # ***************************************************************

    def state(self):
        return calculate_state(
            self_position=self.agent.position,
            common_observation=self.common_observation,
            setting=self.setting,
            synthetic_array=self.synthetic_array,
            blinky_p=self.blinky.position,
            inky_p=self.inky.position
        )

    def update_game_status(self):
        self.process.time += 1

    def agent_move(self, **kwargs):
        self.agent.propose_movement(
            synthetic_array=self.synthetic_array,
            setting=self.setting,
            common_observation=self.common_observation,
            blinky_p=self.blinky.position,
            inky_p=self.inky.position,
            state=self.state(),
            **kwargs
        )
        self.compute_current_reward(self.agent.target_proposal)
        self.maze.validate_proposal(self.agent.target_proposal)
        if self.maze.agent_movement_validation:
            self.agent.last_position = self.agent.position
            self.agent.position = self.agent.target_proposal
        self.process.reward += self.process.current_reward

    def ghost_move(self):
        for ghost in [self.blinky, self.inky]:
            ghost.propose_movement(
                synthetic_array=self.synthetic_array,
                setting=self.setting,
                common_observation=self.common_observation,
                name=ghost.name
            )
            ghost.last_position = ghost.position
            ghost.position = ghost.target_proposal

    def everyone_move(self, **kwargs):
        self.agent_move(**kwargs)
        self.ghost_move()

    # Terminate game if agent collides with ghosts
    def collision_check(self):
        return np.all(self.agent.position == self.blinky.position) or np.all(
            self.agent.position == self.inky.position)

    # Terminate game if agent ghost position swap
    def swap_check(self):
        agent_blinky_swap = np.all(self.agent.last_position == self.blinky.position) and \
                            np.all(self.agent.position == self.blinky.last_position)
        agent_inky_swap = np.all(self.agent.last_position == self.inky.position) and \
                          np.all(self.agent.position == self.inky.last_position)
        return agent_blinky_swap or agent_inky_swap

    # If agent is in training, agent will continue to play even if they meets ghost. This is to explore as much as
    # possible. Therefore, collision check and swap check is only for evaluation game (non-training).
    def consolidate_check(self):
        for check in [self.collision_check, self.swap_check]:
            if check():
                # self.process.current_reward = -30
                return True
        return False

    def check_termination(self):
        # if self.consolidate_check() or self.process.time == self.setting.maximum_time or self.process.win:
        #     # self.process.termination = True
        #     tmp = True
        # else:
        #     tmp = False
        # **********************************************************************************
        if np.sum(self.dots.array) == 0:
            self.process.termination = True
            self.process.win = True
        if self.process.time == self.setting.maximum_time or \
                np.all(self.agent.position == self.blinky.position) or \
                np.all(self.agent.position == self.inky.position) or \
                (tuple(self.blinky.last_position), tuple(self.agent.last_position)) == (
                tuple(self.agent.position), tuple(self.blinky.position)) or \
                (tuple(self.inky.last_position), tuple(self.agent.last_position)) == (
                tuple(self.agent.position), tuple(self.inky.position)):
            self.process.current_reward = self.setting.reward_dict[self.setting.blinky_color]
            self.process.reward += self.process.current_reward
            self.process.termination = True

        # **********************************************************************************

        #
        # if tmp != self.process.termination:
        #     print(tmp, self.process.termination)

    def run(self, **kwargs):  # equivalent to step/move
        """main game running function"""
        self.update_game_status()
        self.everyone_move(**kwargs)

    def update_synthetic_graph_array(self):
        """synthesize maze.array and characters' position to create a new graphic
        array to be rendered by plt"""

        # create maze copy
        array = self.maze.array.copy()

        # delete the dot eaten by agent
        self.dots.array[tuple(self.agent.position)] = 0

        # draw the remaining dots
        array[self.dots.mask()] = self.dots.color
        # draw characters
        array[tuple(self.agent.position)] = self.setting.agent_color
        for ghost in [self.blinky, self.inky]:
            array[tuple(ghost.position)] = getattr(self.setting, ghost.name + '_color')

        # assign new frame array
        self.synthetic_array = array

    def run_one_step_without_graph(self, **kwargs):
        if not self.process.termination:
            self.run(**kwargs)
            self.update_synthetic_graph_array()
            self.common_observation.update(
                self.synthetic_array,
                agent_position=self.agent.position,
                blinky_position=self.blinky.position,
                inky_position=self.inky.position
            )
            # self.compute_current_reward(self.agent.position)
            self.check_termination()

    def run_single_frame(self, **kwargs):
        if not self.process.termination:
            self.run_one_step_without_graph(**kwargs)
            self.graphic.frame.set_data(self.synthetic_array)
        return self.graphic.frame_figure

    def run_and_update_frame(self, frame):
        # this condition check is only used for generating the first frame
        # as there is no need to self.run() for the first frame
        if frame != 0:
            self.run_single_frame()
        sys.stdout.write("\r" + f'Generated frames: {frame}')
        sys.stdout.flush()
        return self.graphic.frame,

    def frame_generator(self):
        """if not terminated keep generating new frame"""
        frame = 0
        while not self.process.termination:
            yield frame
            frame += 1

    def background(self):
        return self.graphic.bg_frame,

    def generate_animation(self):
        self.animation = FuncAnimation(
            self.graphic.frame_figure,
            self.run_and_update_frame,
            init_func=self.background,
            frames=self.frame_generator,
            blit=True,
            save_count=999
        )

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
        self.dots = Dots(self.setting, self.maze.array)

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
