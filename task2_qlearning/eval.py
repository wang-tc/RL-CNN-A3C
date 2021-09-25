import functools
import matplotlib.pyplot as plt
from types import MethodType
from task2_qlearning.q_policy import q_policy


# goals of functions in this file are in the name of the function

def check_win_rate(game, state_func, q_table, total_games):
    policy = functools.partial(q_policy, table=q_table)
    game.state = MethodType(state_func, game)
    game.agent.policy = policy
    win = 0
    for i in range(total_games):
        game.random_reset()
        while not game.process.termination:
            # print(np.sum(game.dots.array))
            game.run_one_step_without_graph()
        if game.process.win:
            win += 1
    return win / total_games


def plot_data(data, path, title, x_name, y_name, fontsize=18):
    fig = plt.figure()
    plt.plot(data)
    fig.suptitle(title, fontsize=fontsize)
    plt.xlabel(x_name, fontsize=fontsize)
    plt.ylabel(y_name, fontsize=fontsize)
    fig.savefig(path)


def prepare_animation(game, policy, state_func, table):
    game.random_reset()
    policy = functools.partial(policy, table=table)
    game.state = MethodType(state_func, game)
    game.agent.policy = policy
    return game
