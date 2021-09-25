import sys
import numpy as np
from types import MethodType
from numpy.random import default_rng
from task1_environment.policy.constants import ACTION_SPACE
from task2_qlearning.q_policy import q_policy


# training script

def train(game, state_func, q_table, total_episode, epsilon, alpha, gamma, evaluate_every, games_per_eval):
    rng = default_rng(1)
    game.setting.maximum_time = 300

    # change default state function
    game.state = MethodType(state_func, game)

    # evaluation
    rewards = []
    eval_reward = []
    eval_living_time = []
    eval_win = []

    # training loop
    for episode in range(total_episode):

        game.random_reset()
        while not game.process.termination:
            s = game.state()
            dice = rng.random()
            if dice < epsilon:
                a = tuple(rng.choice(ACTION_SPACE))
            else:
                a = q_policy(s, q_table)
            game.run_one_step_without_graph(instruction=a)
            r = game.process.current_reward
            ns = game.state()
            max_value = max([q_table[ns, act] for act in ACTION_SPACE])
            ori = q_table[s, a]
            q_table[s, a] = ori + alpha * (r + gamma * max_value - ori)

        # evaluation with 100 games mean

        if episode % evaluate_every == 0:
            eval_r = []
            eval_liv_t = []
            eval_w = 0
            for i in range(games_per_eval):
                game.random_reset()
                while not game.process.termination:
                    s = game.state()
                    a = q_policy(s, q_table)
                    game.run_one_step_without_graph(instruction=a)
                total_reward = game.process.current_reward
                liv_t = game.process.time
                if_win = 1 if game.process.win else 0
                eval_r.append(total_reward)
                eval_liv_t.append(liv_t)
                eval_w += if_win

            eval_reward.append(np.mean(eval_r))
            eval_living_time.append(np.mean(eval_liv_t))
            eval_win.append(eval_w)
        rewards.append(game.process.reward)
        sys.stdout.write("\r" + f'training progress: {((episode+1) / total_episode):.1%}')
        sys.stdout.flush()
    return rewards, eval_reward, eval_living_time, eval_win


# not in use
# # training function
# def train(game, alpha=0.01, gamma=0.99):
#     q_table = game.agent.policy.q_table
#
#     # game.is_training = False
#     # game.agent.policy.random_mode = True
#     previous_state, done = get_initial_status(game)
#     # print('ps', previous_state)
#     # print(game.agent.policy.random_mode)
#     while not done:
#
#         assert(game.agent.policy.random_mode == True)
#         action, state, reward, done = training_step(game)
#
#         max_state_qval = \
#             max(q_table[state, act] for act in [(0, 1), (1, 0), (-1, 0), (0, -1)])
#         q_table[previous_state, action] = \
#             q_table[previous_state, action] + alpha * (
#                     reward + gamma * max_state_qval - q_table[previous_state, action]
#             )
#         # q_table[previous_state, action] = \
#         #     q_table[previous_state, action] + alpha * (
#         #             reward - q_table[previous_state, action]
#         #     )
#         previous_state = state
#     return game.process.reward

# win = 0
# for episode in range(100):
#     game.random_reset()
#     while not game.process.termination:
#         s = game.state()
#         a = q_policy(s, q_table)
#         game.run_one_step_without_graph(instruction=a)
# #     print(game.process.reward)
#     if game.process.win:
#         win+=1
# print(win)