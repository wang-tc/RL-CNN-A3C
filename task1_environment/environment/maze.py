import numpy as np


class Maze:
    def __init__(self, setting):

        self.wall_color = setting.wall_color
        self.path_color = setting.path_color

        # create array
        self.array = np.zeros((
            setting.maze_height, setting.maze_width
        ))

        # get height
        # self.x_len = self.array.shape[0]

        # get width
        # self.y_len = self.array.shape[1]

        # create border
        self.array[0, :] = \
            self.array[setting.maze_height - 1, :] = \
            self.array[:, 0] = \
            self.array[:, setting.maze_width - 1] = self.wall_color

        # create tunnel
        self.array[1:setting.maze_height - 1:setting.maze_row_height, 1:setting.maze_width - 1] = \
            self.array[1:setting.maze_height - 1, 1:setting.maze_width - 1:setting.maze_column_width] = \
            setting.path_color

        # create block
        self.array[(self.array != setting.path_color)] = self.wall_color

        self.agent_movement_validation = None

    def validate_proposal(self, proposal):
        if self.array[tuple(proposal)] == self.wall_color:
            self.agent_movement_validation = False
        else:
            self.agent_movement_validation = True


class Dots:
    def __init__(self, setting, array):
        self.color = setting.dot_color
        self.array = array.copy()
        mask = self.array == setting.path_color
        self.array[mask] = self.color
        self.array[~mask] = 0

    def mask(self):
        return self.array == self.color


