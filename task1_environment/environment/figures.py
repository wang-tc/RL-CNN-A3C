import numpy as np

# setting class control the general settings of the game, such as size, colors, maximum time step
# Process class record the key figures and states of the game such as time, reward, if_win and
# if_terminate.


class Setting:
    def __init__(
            self, maze_row_height=10,
            maze_column_width=10, maze_row_num=5, maze_column_num=15,
            graphic_size=(30, 10)
    ):
        # maze
        self.maze_row_num = maze_row_num
        self.maze_column_num = maze_column_num
        self.maze_row_height = maze_row_height
        self.maze_column_width = maze_column_width
        self.maze_height = self.maze_row_height * self.maze_row_num + 3
        self.maze_width = self.maze_column_width * self.maze_column_num + 3
        self.graphic_figsize = graphic_size
        self.path_color = 10
        self.wall_color = 0

        # character

        # agent
        self.agent_color = 6
        self.agent_init_position = np.array((1, 1))

        # ghost
        self.blinky_init_position = np.array((1, self.maze_width - 2))
        self.blinky_color = 12

        self.inky_init_position = np.array((self.maze_height - 2, self.maze_width - 2))
        self.inky_color = 11
        # self.inky_color = 12

        # dot
        self.dot_color = 3
        # self.dot_color = 10

        # default maximum time set to 5
        # if want to end game sooner, please use:
        # game = Pacman()
        # game.setting.maximum_time = shorter_time
        self.maximum_time = 2000

        # reward
        # for task 4
        # self.reward_dict = {
        #     self.dot_color: 10,
        #     self.path_color: -1,
        #     self.blinky_color: -20,
        #     self.inky_color: -20,
        #     self.wall_color: -5,
        # }

        # best for task 2
        self.reward_dict = {
            self.dot_color: 10,
            self.path_color: -1,
            self.blinky_color: -99,
            self.inky_color: -99,
            self.wall_color: -20,
        }

        # for task 4
        # self.reward_dict = {
        #     self.dot_color: 1,
        #     self.path_color: -0.01,
        #     self.blinky_color: -1,
        #     self.inky_color: -1,
        #     self.wall_color: -0.1,
        # }


class Process:
    def __init__(self):
        self.time = 0
        self.reward = 0
        self.current_reward = 0
        self.termination = False
        self.win = False

    def reset(self):
        self.__init__()
