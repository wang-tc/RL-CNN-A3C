import numpy as np
from numpy import linalg as LA
# from task4_a3c_cnn.train import f3x3_vision, view_self
from task4_a3c_cnn.utils import v_wrap

ACTION_MAP = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1)
}


# keep 3x3 surrounding cells and mask everything else
def f3x3_vision(env, name):
    # array = np.pad(array, pad_width=1, mode='constant',
    #                constant_values=0)
    array = env.synthetic_array.copy()
    character = getattr(env, name)
    p = character.position
    x, y = p
    array[x - 1:x + 2, y - 1:y + 2] *= -1
    array[array > 0] = 0
    array[x - 1:x + 2, y - 1:y + 2] *= -1
    return array


# keep self cell and mask every thing else
def view_self(env, name):
    array = env.synthetic_array.copy()
    character = getattr(env, name)
    # p = character.position
    color = getattr(env.setting, name + '_color')
    array[array != color] = 0
    array[array == color] = 1
    return array


# produce 9 images explained in report
def visualize(game, HEIGHT):
    agent_view = f3x3_vision(game, 'agent')
    blinky_view = f3x3_vision(game, 'blinky')
    inky_view = f3x3_vision(game, 'inky')
    agent = view_self(game, 'agent')
    blinky = view_self(game, 'blinky')
    inky = view_self(game, 'inky')
    dot = game.synthetic_array.copy()
    dot[dot != game.setting.dot_color] = 0
    dot[dot != 0] = 1
    path = game.synthetic_array.copy()
    path[path != game.setting.path_color] = 0
    path[path != 0] = 1
    chas = game.synthetic_array.copy()
    chas[((chas != game.setting.agent_color) &
          (chas != game.setting.blinky_color) &
          (chas != game.setting.inky_color))] = 0
    features = [
        agent_view, blinky_view, inky_view, agent, blinky, inky, dot, path, chas
    ]
    features = np.stack(features, axis=0)
    features += 0.1
    # norm = LA.norm(features)
    features /= 14
    features = features.reshape((1, 9, HEIGHT, HEIGHT))
    return features


# task 4 compatible policy
def channel_9_policy(game, height, model, **kwargs):
    s = visualize(game, height)
    a = model.choose_action(v_wrap(s))
    action = ACTION_MAP[a]
    return action


# not in use, previous policy
class A3CTwo:
    def __init__(self, model):
        self.action_map = ACTION_MAP
        self.lnet = model
        self.lnet.eval()

    def __call__(self, **policy_kwargs):
        array = policy_kwargs['synthetic_array']
        self.lnet.eval()
        a = self.lnet.choose_action(v_wrap(array[None, None, :]))
        # print('\n',a)
        action = ACTION_MAP[a]

        return action
