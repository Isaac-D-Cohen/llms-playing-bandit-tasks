from random import randint

class Random:
    def __init__(self, name):
        self.name = name

    def change_name(self, name):
        self.name = name

    def setup(self, labels, arms, T, bernoulli):
        self.arms = arms

    def update(self, chosen_arm, a, reward):
        pass

    def make_decision(self):
        rand = randint(0, len(self.arms)-1)
        return (self.arms[rand], "N/A")

    def get_params(self):
        params = dict()

        params['type'] = "Random"
        params['name'] = self.name

        return params
