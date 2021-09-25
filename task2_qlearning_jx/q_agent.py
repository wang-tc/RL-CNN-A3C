import sys
sys.path.append("/Deep-Reinforcement-Learinng-Submission/task1_environment/environment")
import numpy as np
import random
from task1_environment.environment.character import Character
from task1_environment.policy.baseline import observe

rng = np.random.default_rng()


class Agent(Character):
    def __init__(self, name, position, policy):
        super().__init__(position, policy)
        self.name = name
        self.epsilon = 0.05  # Exploration rate
        self.gamma = 0.99  # Discount factor
        self.alpha = 0.01  # Learning rate
        self.Q_values = {}
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.epsilon_greedy = False
        self.net = None
        # self.dots_left = None

    def epsilon_greedy_choose(self):
        """Return an action to try in this state."""
        p = random.random()
        if p < self.epsilon:
            self.epsilon_greedy = True
        else:
            self.epsilon_greedy = False

    def Q(self, s, a):
        s = tuple(s)
        """Return the estimated Q-value of this action in this state."""
        if (s, a) not in self.Q_values:
            for act in self.actions:
                self.Q_values[(s, act)] = 1
        return self.Q_values[(s, a)]

    # def observe(self, s, a, sp, r):
    #     s = str(s)
    #     # old_sa = self.Q(s, a)
    #     max_value = max([self.Q(sp, a) for a in self.actions])
    #     self.Q_values[(s, a)] = self.Q(s, a) + self.alpha * (r + self.gamma * max_value - self.Q(s, a))
    #     # print(self.Q_values[(s, a)])
    #     # return self.Q_values[(s, a)]

    def propose_movement(self, **kwargs):
        if self.epsilon_greedy:
            self.direction_proposal = random.choice(self.actions)
            self.target_proposal = self.position + self.direction_proposal
        else:
            ob_kwargs = {'self_position': self.position, **kwargs}
            self.observation = self.observe(**ob_kwargs)
            policy_kwargs = {'observation': self.observation,
                             'actions': self.actions,
                             'Q': self.Q,
                             'net':self.net,**kwargs}
            self.direction_proposal = self.policy(**policy_kwargs)
            self.target_proposal = self.position + self.direction_proposal
        self.epsilon_greedy = False
