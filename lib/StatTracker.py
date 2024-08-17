import numpy as np
import pandas as pd

class StatTracker:
    # best_arm is the name of the best arm
    def __init__(self, solver_name, K, best_arm):

        self.solver_name = solver_name

        # to keep track of stats
        self.K = K
        self.data = []
        self.Q = np.zeros(K)    # tracks the current action values of the arms
        self.N = np.zeros(K)    # tracks how many times an arm is chosen
        self.total_reward = 0
        self.best_arm = best_arm
        self.greedy_times = 0       # this will hold the raw number of times the greedy arm is chosen
        self.suffix = 0             # this will hold the last step on which best_arm is chosens
        self.percent_optimal = 0
        self.n_invalid = 0

    # a is the chosen arm index, reward is what we got
    def update_stats(self, step, response, chosen_arm, a, reward, valid):

        self.total_reward += reward

        # was the true best arm chosen?
        if chosen_arm == self.best_arm:
            chose_best = 1
            self.suffix = step
        else:
            chose_best = 0


        # was the arm with the currently highest action value chosen?
        chose_maxQ = 1 if self.Q[a] == max(self.Q) else 0
        self.greedy_times += chose_maxQ

        # update action values
        self.N[a] = self.N[a] + 1
        self.Q[a] = self.Q[a] + (1 / self.N[a]) * (reward - self.Q[a])

        self.percent_optimal = self.percent_optimal + (1 / (step+1)) * (chose_best - self.percent_optimal)

        # get k*min_frac(R, t) for this replicate
        k_min_frac = self.K * self.N.min()/(step+1)

        # record data
        step_data = [step, self.best_arm, response, chosen_arm, valid, chose_best, chose_maxQ, reward, self.total_reward/(step+1), self.percent_optimal, k_min_frac]
        self.data.append(step_data)

        if valid == False:
            self.n_invalid += 1

    # the following two functions should be read as const
    # they allow you to enter the index of an arm and get
    # whether selecting it would be greedy or optimal
    def is_greedy(self, a):
        return True if self.Q[a] == max(self.Q) else False

    def is_anti_greedy(self, a):
        return True if self.Q[a] == min(self.Q) else False

    def is_optimal(self, chosen_arm):
        return True if chosen_arm == self.best_arm else False

    def get_dataframe(self):
        return pd.DataFrame(self.data, columns=['step','best_arm','response','chosen_arm','valid','optimal','max_Q','reward', 'average reward', '% optimal', 'k*min_frac'])

    def pickle_it(self, filepath):
        frame = pd.DataFrame(self.data, columns=['step','best_arm','response','chosen_arm', 'valid', 'optimal','max_Q','reward', 'average reward', '% optimal', 'k*min_frac'])
        frame.to_pickle(filepath)

    # we can add more summary statistics and show them here (like greedy fraction etc)
    def get_summary(self, T):
        return {"Solver name:": self.solver_name, "Total reward:": self.total_reward, "Greedy Frac:": float(self.greedy_times)/T, "Suffix Fail (T/2):": self.suffix < T/2, "Suffix point:": (self.suffix+1)/T, 'Invalid:': self.n_invalid}
