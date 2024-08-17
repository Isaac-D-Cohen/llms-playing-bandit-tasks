from tqdm import tqdm
import numpy as np
from lib.StatTracker import StatTracker
from lib.Solvers.UCB import UCB
from random import randint
import pandas as pd

class MAB:

    def __init__(self, K=5, T=100, gap=0.2, seed=None, bernoulli=True, mean=0.5, stdev=1):

        # random number generator
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        self.mean = mean
        self.stdev = stdev       # only matters for bernoulli = False (i.e. Gaussian)

        # mean rewards
        self.mu = np.array(
            [self.mean + gap/2.0] +
            [self.mean - gap/2.0] * (K - 1)
        )

        self.K = K
        self.T = T
        self.gap = gap
        self.bernoulli = bernoulli


    def fresh_instance(self):

        # arms (randomized)
        if self.K == 5:
            self.arms = np.array(['blue','green','purple','red','yellow'])
            self.labels = "blue, green, red, yellow, purple"
        elif self.K == 4:
            self.arms = np.array(['blue','green','red','yellow'])
            self.labels = "blue, green, red, yellow"
        elif self.K == 3:
            self.arms = np.array(['blue','red','yellow'])
            self.labels = "blue, red, yellow"
        elif self.K == 2:
            self.arms = np.array(['blue','red'])
            self.labels = "blue, red"

        # shuffle the means of the arms
        self.rng.shuffle(self.mu)

        # best arm will store the color of the best arm
        self.best_arm = self.arms[np.argmax(self.mu)]

        # indices for each arm
        self.arm_ids = dict(zip(self.arms, range(self.K)))


    def simulate(self, solvers):

        assert self.arms is not None

        stats = []
        rewards = np.empty(len(self.arms))

        for solver in solvers:
            # make a StatTracker instance for each solver
            stats.append(StatTracker(solver.name, self.K, self.best_arm))

            # setup the solver
            solver.setup(self.labels, self.arms, self.T, bernoulli=self.bernoulli)


        for step in tqdm(range(self.T)):

            # draw from each arm's reward distribution
            # if bernoulli, do a binomial
            if self.bernoulli == True:
                for i in range(len(self.arms)):
                    rewards[i] = self.rng.binomial(n=1, p=self.mu[i])
            # otherwise do a normal
            else:
                for i in range(len(self.arms)):
                    rewards[i] = self.rng.normal(loc=self.mu[i], scale=self.stdev)

            for i in range(len(solvers)):
                (chosen_arm, response) = solvers[i].make_decision()

                if chosen_arm in self.arm_ids:
                    # index of chosen arm
                    a = self.arm_ids[chosen_arm]
                    valid = True
                else:
                    # LLM did not choose a valid arm, take a random one (sadly)
                    a = randint(0, len(self.arms)-1)
                    chosen_arm = self.arms[a]
                    valid = False

                # get the reward for our chosen arm
                reward = rewards[a]

                # update the solver's history with choice and reward
                solvers[i].update(chosen_arm, a, reward)

                stats[i].update_stats(step, response, chosen_arm, a, reward, valid)

        return stats


    # run a simulation where, up to a given trial one solver generates the history
    # and then another solver just gives its answer for trial k
    # we do this n times, and then the code calling us can make a new MAB instance
    # we record whether the selected option is greedy and whether it's the best option
    # (k*min_frac does not make sense for one trial - in the future we can add
    # code that records the UCB ranking of the arm the model chose)
    # deciders and history_makers must have the same number of elements - solvers
    # note: there is only a purpose in setting n > 1, if your decider or history_maker is not deterministic
    # replicate_number is just taken for the purpose of naming the outputs

    def run_alternate_simulation(self, deciders, history_makers, n, k, replicate_number):

        assert self.arms is not None
        assert len(deciders) == len(history_makers)

        rewards = np.empty(len(self.arms))
        stat_tracker_array = []
        summary_array = []          # we make our own summaries because the ones StatTracker makes aren't particularly useful for this task
        dataframe_array = []        # this will store the choices each decider makes

        for solver in deciders:
            summary_array.append({'Name': solver.name + ' #' + str(replicate_number), 'Greedy': 0.0, 'AntiGreedy': 0.0, 'Optimal': 0.0, 'Invalid': 0})
            dataframe_array.append([])

        for r in tqdm(range(n)):

            # for each of these rounds we will make an array like this and concatenate
            # with the big one at the end of this loop
            iteration_stat_tracker_array = []

            for solver in deciders:
                solver.setup(self.labels, self.arms, self.T, bernoulli=self.bernoulli)
                iteration_stat_tracker_array.append(StatTracker(solver.name + ' - ' + str(r) + ' #' + str(replicate_number), self.K, self.best_arm))

            for solver in history_makers:
                solver.setup(self.labels, self.arms, self.T, bernoulli=self.bernoulli)

            for step in range(k-1):

                # draw from each arm's reward distribution
                # if bernoulli, do a binomial
                if self.bernoulli == True:
                    for i in range(len(self.arms)):
                        rewards[i] = self.rng.binomial(n=1, p=self.mu[i])
                # otherwise do a normal
                else:
                    for i in range(len(self.arms)):
                        rewards[i] = self.rng.normal(loc=self.mu[i], scale=self.stdev)

                # now let's make history for this trial
                for i in range(len(history_makers)):
                    # get each history maker's decision
                    chosen_arm, response = history_makers[i].make_decision()

                    if chosen_arm in self.arm_ids:
                        # index of chosen arm
                        a = self.arm_ids[chosen_arm]
                        valid = True
                    else:
                        # history maker did not choose a valid arm, take a random one (sadly)
                        a = randint(0, len(self.arms)-1)
                        chosen_arm = self.arms[a]
                        valid = False

                    # get the reward for our chosen arm
                    reward = rewards[a]

                    # update the history maker and decider's history with choice and reward
                    history_makers[i].update(chosen_arm, a, reward)
                    deciders[i].update(chosen_arm, a, reward)

                    # update the dataframe
                    iteration_stat_tracker_array[i].update_stats(step, response, chosen_arm, a, reward, valid)


            # ok, now we have all our deciders loaded with history from our history makers
            # now let's ask them to make a decision
            for i in range(len(deciders)):
                chosen_arm, response = deciders[i].make_decision()

                # iterpret the decider's answer
                if chosen_arm in self.arm_ids:
                    # index of chosen arm
                    a = self.arm_ids[chosen_arm]
                    invalid = False
                else:
                    # decider did not choose a valid arm, take a random one (sadly)
                    a = randint(0, len(self.arms)-1)
                    chosen_arm = self.arms[a]
                    invalid = True

                greedy = iteration_stat_tracker_array[i].is_greedy(a)
                anti_greedy = iteration_stat_tracker_array[i].is_anti_greedy(a)
                optimal = iteration_stat_tracker_array[i].is_optimal(a)

                # update the summary array - note that python let's us add booleans to floats
                summary_array[i]['Greedy'] += greedy
                summary_array[i]['AntiGreedy'] += anti_greedy
                summary_array[i]['Optimal'] += optimal
                summary_array[i]['Invalid'] += invalid

                dataframe_array[i].append([r, self.best_arm, response, chosen_arm, greedy, anti_greedy, optimal, invalid])

            stat_tracker_array += iteration_stat_tracker_array

        for summary in summary_array:
            summary['Greedy'] /= n
            summary['AntiGreedy'] /= n
            summary['Optimal'] /= n

        for i in range(len(deciders)):
            dataframe_array[i] = pd.DataFrame(dataframe_array[i], columns=["Round", "Best Arm", "Response", "Chosen Arm", "Greedy", "AntiGreedy", "Optimal", "Invalid"])

        return (stat_tracker_array, summary_array, dataframe_array)


    # we will go from 0 to step k-1
    # the history_makers will make histroy, the deciders will decide on every move
    # we will record both, history as created by the history makers and,
    # in a separate dataframe, the move chosen by each decider along with records of
    # the UCB values
    # in the future we can add options to change this to greedy values or what-else-not
    # for now this only works with UCB history_makers
    def run_alternate_simulation2(self, deciders, history_makers, k, replicate_number):

        assert self.arms is not None
        assert len(deciders) == len(history_makers)

        rewards = np.empty(len(self.arms))

        history_trackers = []
        ucb_value_trackers = []

        for solver in deciders:
            solver.setup(self.labels, self.arms, self.T, bernoulli=self.bernoulli)
            ucb_value_trackers.append([])

        for solver in history_makers:
            # for now this only works with UCB history_makers
            assert isinstance(solver, UCB)

            solver.setup(self.labels, self.arms, self.T, bernoulli=self.bernoulli)
            history_trackers.append(StatTracker(solver.name + ' #' + str(replicate_number), self.K, self.best_arm))

        for step in tqdm(range(k)):

            # draw from each arm's reward distribution
            # if bernoulli, do a binomial
            if self.bernoulli == True:
                for i in range(len(self.arms)):
                    rewards[i] = self.rng.binomial(n=1, p=self.mu[i])
            # otherwise do a normal
            else:
                for i in range(len(self.arms)):
                    rewards[i] = self.rng.normal(loc=self.mu[i], scale=self.stdev)

            # go through history makers and deciders
            for i in range(len(deciders)):

                # decider decides, but we don't care what
                # someday, if needed, we can modify this to record the
                # deciders' answers
                _, _ = deciders[i].make_decision()

                # now the answer we actually care about
                chosen_arm, response = history_makers[i].make_decision()

                # someday if we decide to allow non-UCB history_makers
                # we would need this code because the solver might potentially
                # give an unintelligible answer

                # if chosen_arm in self.arm_ids:
                #     # index of chosen arm
                #     a = self.arm_ids[chosen_arm]
                #     valid = True
                # else:
                #     # history maker did not choose a valid arm, take a random one (sadly)
                #     a = randint(0, len(self.arms)-1)
                #     chosen_arm = self.arms[a]
                #     valid = False

                # for now...
                a = self.arm_ids[chosen_arm]
                valid = True

                # get the reward for our chosen arm
                reward = rewards[a]

                # now before appending the answer and moving on
                # let's store the current UCB values
                # remember that all history_makers are UCB instances
                greedy_values, ucb_components = history_makers[i].get_action_values()

                ucb_value_trackers[i].append(list(greedy_values) + list(ucb_components))

                # update the history maker and decider
                history_makers[i].update(chosen_arm, a, reward)
                deciders[i].update(chosen_arm, a, reward)
                history_trackers[i].update_stats(step, response, chosen_arm, a, reward, valid)


        columns_a = []
        columns_b = []
        for i in range(self.K):
            columns_a.append(f"Arm {i} Q(a)")
            columns_b.append(f"Arm {i} UCB_Factor")

        for i in range(len(deciders)):
            ucb_value_trackers[i] = pd.DataFrame(ucb_value_trackers[i], columns = columns_a+columns_b)

        return (history_trackers, ucb_value_trackers)


    def get_params(self):

        params = dict()

        params['seed'] = self.seed
        params['mean'] = self.mean
        params['stdev'] = self.stdev
        params['K'] = self.K
        params['T'] = self.T
        params['gap'] = self.gap
        params['bernoulli'] = self.bernoulli

        return params

