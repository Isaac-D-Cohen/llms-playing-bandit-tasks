import numpy as np

class TS:
    def __init__(self, name):
        self.name = name

    def change_name(self, name):
        self.name = name

    def setup(self, labels, arms, T, bernoulli):
        self.arms = arms
        self.alpha = np.ones(len(arms))
        self.beta = np.ones(len(arms))

    def update(self, chosen_arm, a, reward):
        if reward == 1:
            self.alpha[a] += 1
        else:
            self.beta[a] += 1

    def make_decision(self):
        sampled_values = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(len(self.arms))]
        max_index = np.argmax(sampled_values)
        return (self.arms[max_index], "N/A")

    def get_params(self):
        params = dict()

        params['type'] = "TS"
        params['name'] = self.name

        return params
