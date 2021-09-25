import os
import torch as T
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


from task2_qlearning_jx.main import PacManJX
from task3_a3c_ann.a3c_policy import A3C_Policy

# from A3C.optimizer import SharedAdam



class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0.0001):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims, 64)
        self.v1 = nn.Linear(*input_dims, 64)

        self.pi2 = nn.Linear(64, 64)
        self.v2 = nn.Linear(64, 64)

        self.pi = nn.Linear(64, n_actions)
        self.v = nn.Linear(64, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        pi1 = torch.tanh(self.pi1(state))
        pi2 = torch.tanh(self.pi2(pi1))
        v1 = torch.tanh(self.v1(state))
        v2 = torch.tanh(self.v2(v1))

        pi = self.pi(pi2)
        v = self.v(v2)

        return pi, v

    def calc_R(self, win):
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states)

        R = v[-1] * (1 - int(win))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    def calc_loss(self, win):
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R(win)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns - values) ** 2

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs * (returns - values)

        total_loss = (critic_loss + actor_loss).mean()

        return total_loss

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        pi, v = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]
        # print(dist.sample())

        return action


class Workers(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions,
                gamma, lr, name, global_ep_idx, episode_num):
        super(Workers, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        # set pacman environment here
        self.env = PacManJX(maze_row_num=2, maze_column_num=2, maze_row_height=2, maze_column_width=2)
        self.env.maximum_time = 300
        self.optimizer = optimizer
        self.episode_num = episode_num

    def run(self):
        while self.episode_idx.value < self.episode_num:
            self.env.random_reset()
            # observation = self.env.common_observation.agent_three_square.flatten().tolist() ######
            observation = self.env.state()
            # print(state)
            curr_dots = self.env.dots.curr_dotsNum
            # observation.append(curr_dots)

            score = 0
            self.local_actor_critic.clear_memory()
            while not self.env.process.termination:
                action = self.local_actor_critic.choose_action(observation) ######
                self.env.agent.policy = A3C_Policy(action).policy
                # print(action,self.env.agent.direction_proposal)


                # observation_ = self.env.common_observation.agent_three_square.flatten().tolist()
                # print(self.env.agent.position)
                # observation_.append(curr_dots)

                # observation_ = self.env.step(action)  ######
                self.env.run_one_step_without_graph()
                observation_ = self.env.state()
                reward = self.env.process.current_reward

                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                # print(self.env.process.reward,score)

                if self.env.process.termination or self.env.process.time % 5 == 0:
                    loss = self.local_actor_critic.calc_loss(self.env.process.win)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    # print('opt',self.optimizer.state)
                    self.local_actor_critic.load_state_dict(
                        self.global_actor_critic.state_dict())
                    # print(self.local_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                observation = observation_ #####
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.env.process.win)
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    load_path = 'saved_models/a3c_ann_lr1e-4_gamma_0.9.pt'
    save_path = 'saved_models/a3c_ann_lr1e-4_gamma_0.9.pt'
    lr = 1e-4
    n_actions = 4
    input_dims = [15]
    a = ActorCritic(input_dims, n_actions)
    global_actor_critic = a
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr,
                        betas=(0.92, 0.999))
    try:
        state_dict = torch.load(load_path)
        global_actor_critic.load_state_dict(state_dict['model_state_dict'])
        global_actor_critic.load_state_dict(state_dict['optimizer_state_dict'])
    except FileNotFoundError:
        print('new model')

    global_ep = mp.Value('i', 0)

    workers = [Workers(global_actor_critic,
                    optim,
                    input_dims,
                    n_actions,
                    gamma=0.9,
                    lr=lr,
                    name=i,
                    global_ep_idx=global_ep,
                    episode_num = 100000) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]

    state_dict = {
        'model_state_dict': global_actor_critic.state_dict(),
        'optimizer_state_dict': global_actor_critic.state_dict(),
    }
    if not os.path.isfile(save_path) or input('Checkpoint exist, return 1 to overwrite.') == "1":
        torch.save(state_dict, save_path)
    else:
        print('Model not saved.')

    # plt.plot(workers.)
    # plt.show()

    # lr = 1e-3
    # n_actions = 4
    # input_dims = [15]
    # global_actor_critic = ActorCritic(input_dims, n_actions)
    # global_actor_critic.share_memory()
    # optim = SharedAdam(global_actor_critic.parameters(), lr=lr,
    #                    betas=(0.92, 0.999))
    # global_ep = mp.Value('i', 0)
    #
    # a = Workers(global_actor_critic,
    #                    optim,
    #                    input_dims,
    #                    n_actions,
    #                    gamma=0.99,
    #                    lr=lr,
    #                    name=0,
    #                    global_ep_idx=global_ep,
    #                    episode_num=100)
    # a.run()








