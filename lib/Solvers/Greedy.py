import numpy as np
from random import random, randint

class Greedy:
    def __init__(self, name, epsilon=0.0, try_each_once=False, initial_action_values=0.0):
        self.name = name
        self.epsilon = epsilon
        self.try_each_once = try_each_once
        self.initial_action_values = initial_action_values

    def change_name(self, name):
        self.name = name

    def setup(self, labels, arms, T, bernoulli):
        self.arms = arms
        self.K = len(arms)
        self.Q = np.full(self.K, self.initial_action_values)    # tracks the current action values of the arms
        self.N = np.zeros(self.K)    # tracks how many times we choose each arm so we can compute the action value
        self.step = 0

    def update(self, chosen_arm, a, reward):
        # update the expected value of the chosen arm with the reward we got
        self.step += 1
        self.N[a] += 1
        self.Q[a] = self.Q[a] + (1 / self.N[a]) * (reward - self.Q[a])

    def make_decision(self):

        # if we were told to try each arm at least once
        if self.step < self.K and self.try_each_once == True:
            chosen_arm = self.step

        else:
            # choose a random number to decide whether to explore
            if random() >= self.epsilon:
                chosen_arm = np.random.choice(np.argwhere(self.Q == np.max(self.Q)).flatten())
            else:
                chosen_arm = randint(0, len(self.arms)-1)

        return (self.arms[chosen_arm], "N/A")

    def get_params(self):

        params = dict()

        params['type'] = "Greedy"
        params['name'] = self.name
        params['epsilon'] = self.epsilon
        params['try_each_once'] = self.try_each_once
        params['initial_action_values'] = self.initial_action_values

        return params
