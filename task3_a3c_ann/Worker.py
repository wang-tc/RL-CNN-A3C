# reference from https://morvanzhou.github.io
import os
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from task1_environment.environment.main import PacMan
from task2_qlearning_jx.main import PacManJX
from task3_a3c_ann.a3c_policy import A3C_Policy

import torch
import torch.nn as nn
from task3_a3c_ann.utils import v_wrap, set_init, push_and_pull
import torch.nn.functional as F
import torch.multiprocessing as mp


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


# os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.95
MAX_EP = 100000

env = PacManJX(maze_row_num=2, maze_column_num=2, maze_row_height=2, maze_column_width=2)
N_S, N_A = 15, 4  # input dimension and output dimension

class Net(nn.Module):
    def __init__(self, s_dim=N_S, a_dim=N_A):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)  # local network
        self.env = PacManJX(maze_row_num=1, maze_column_num=2, maze_row_height=2, maze_column_width=2)
        self.blinky_random = False
        self.env.setting.maximum_time = 400

    def record(self,global_ep, global_ep_r, ep_r, res_queue, name):
        with global_ep.get_lock():
            global_ep.value += 1
        with global_ep_r.get_lock():
            if global_ep_r.value == 0.:
                global_ep_r.value = ep_r
            else:
                global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
        res_queue.put(global_ep_r.value)
        print(
            name,
            "Ep:", global_ep.value,
            "| Ep_r: %.0f" % global_ep_r.value,
        )

    def run(self):
        total_step = 0
        while self.g_ep.value < MAX_EP:
            self.env.random_reset()
            s = self.env.state()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                # print(env.agent.position)
                move = A3C_Policy(a).policy
                self.env.agent.policy = move # update action
                self.env.run_one_step_without_graph()
                s_ = self.env.state()
                r = self.env.process.current_reward
                # print(r)
                # print('{','chuurent',r,'reward','agent',self.env.agent.position,
                #       'blinky',self.env.blinky.position,
                #       'inky',self.env.inky.position,'}')
                # print('{','current',r,'}')
                done = self.env.process.termination

                # if done: ep_r -= r
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    win = self.env.process.win
                    push_and_pull(self.opt, self.lnet, self.gnet, win, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        self.record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
            print(self.env.process.win,self.env.process.time,self.env.process.reward)
        self.res_queue.put(None)


if __name__ == "__main__":
    load_path = 'saved_models/1e_4_0.95.pt'
    save_path = 'saved_models/1e_4_0.95.pt'
    gnet = Net(N_S, N_A)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=5e-4, betas=(0.92, 0.9999))  # global optimizer

    try:
        state_dict = torch.load(load_path)
        gnet.load_state_dict(state_dict['model_state_dict'])
        opt.load_state_dict(state_dict['optimizer_state_dict'])
    except FileNotFoundError:
        print('new model')

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    state_dict = {
        'model_state_dict': gnet.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
    }
    if not os.path.isfile(save_path) or input('Checkpoint exist, return 1 to overwrite.') == "1":
        torch.save(state_dict, save_path)
    else:
        print('Model not saved.')

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
