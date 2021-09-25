import torch
import numpy as np
from torch import nn
import functools


# convert numpy to torch
def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


# initialise weight
def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


# back propagation
def push_and_pull(opt, lnet, gnet, win, done, s_, bs, ba, br, gamma):
    if win:
        v_s_ = 0.  # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:  # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


# show training progress
def record(global_ep, global_ep_r, ep_r, res_queue, name):
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


# as in name
def prepare_animation(game, model, policy, checkpoint_path, height):
    model.eval()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    policy = functools.partial(policy, game=game, height=height, model=model)
    game.agent.set_policy(policy)
    game.random_reset()
    return game


# as in name
def check_winrate(game, total_games):
    win = 0
    for i in range(total_games):
        game.random_reset()
        while not game.process.termination:
            game.run_one_step_without_graph()
        if game.process.win:
            win += 1
    return win / total_games
