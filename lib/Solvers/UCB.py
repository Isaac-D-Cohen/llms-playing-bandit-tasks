from math import log, sqrt, e
import numpy as np

class UCB:
    def __init__(self, name, log_base=e, const=1.0, initial_values=1):
        self.name = name
        self.const = const
        self.initial_values = initial_values
        self.log_base = log_base

    def change_name(self, name):
        self.name = name

    def setup(self, labels, arms, T, bernoulli):
        self.arms = arms
        self.Q = np.array([float(self.initial_values)]*len(arms))
        self.N = np.zeros(len(self.Q))
        self.step = 0

    def update(self, chosen_arm, a, reward):
        self.step += 1
        self.N[a] += 1
        self.Q[a] = self.Q[a] + (1 / self.N[a]) * (reward - self.Q[a])

    def make_decision(self):

        if min(self.N) == 0:
            chosen_arm = np.random.choice(np.argwhere(self.N == 0).flatten())
            return (self.arms[chosen_arm], "N/A")

        # get the value of each arm according to UCB
        action_values = np.empty(len(self.Q))
        for i in range(len(self.Q)):
            action_values[i] = self.Q[i] + self.const*sqrt(log(self.step, self.log_base)/self.N[i])

        # choose an arm with a heighest value
        chosen_arm = np.random.choice(np.argwhere(action_values == np.max(action_values)).flatten())
        return (self.arms[chosen_arm], "N/A")

    # special function that other solvers don't necessarily have
    # we needed it for an experiment, so we created it
    def get_action_values(self):

        ucb_components = np.empty(len(self.Q))
        for i in range(len(self.Q)):
            if self.N[i] == 0:
                ucb_components[i] = float("inf")
            else:
                ucb_components[i] = self.const*sqrt(log(self.step, self.log_base)/self.N[i])

        return (self.Q, ucb_components)

    def get_params(self):

        params = dict()

        params['type'] = "UCB"
        params['name'] = self.name
        params['const'] = self.const
        params['initial_values'] = self.initial_values
        params['log_base'] = self.log_base

        return params
