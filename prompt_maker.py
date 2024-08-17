from lib.Solvers.LLM import LLM
from lib.Solvers.Greedy import Greedy
from lib.Solvers.UCB import UCB
from lib.Solvers.TS import TS
from lib.Solvers.Random import Random
import pandas as pd
import numpy as np
from math import gcd
import json
import os

input_directory = "inputs"
output_directory = "outputs"

# solvers that we use to make the prompt
# we can require that some of them disagree with each other
solver_configurations = [
    {"type": "Greedy", "name": "Greedy e=0", "epsilon": 0.0, "try_each_once": False, "initial_action_values": 0.0},
    {"type": "UCB", "name": "UCB", "const": 1.0, "initial_values": 0, "log_base": 10},
    # {"type": "TS", "name": "Thompson Sampling"},
]

# for a given solver, allow or disallow it to say the same thing as last time
# setting its value to false would make us skip any otherwise acceptable prompt where
# a solver repeats its color
allow_repetition = [False, False, True]

# config for the LLM prompt we are making
# for the time being our code only supports this particular LLM configuration
# the model is technically irrelevant here, except that the prompt will be geared towards this one
LLM_config = {"type": "LLM", "name": "llama3 temp=0.01", "model": "meta-llama/Meta-Llama-3-8B-Instruct", "balance_EE": True, "short_answers": True, "temperature": 0.01, "end_with_goal": True, "save_every_nth_response": 0, "provide_context_on_round": True}

# indicate which solvers must disagree by giving them non-overlapping prime factorizations
# this list has len(solver_configurations)+1 elements because the history dataframe is also a solver
agreement = [2,3,6]


def too_much_agreement(decisions, agreement):

    for i in range(len(decisions)):
        for j in range(len(decisions)):
            # if two deciders agree
            if decisions[i] == decisions[j]:
                # if they don't share any common factors they must disagree
                if gcd(agreement[i], agreement[j]) == 1:
                    return True
    return False


# checks if any of the models says the same thing it did last time
# and whether this is allowed
def too_much_repetition(decisions, allow_repetition, decision_history):
    for i in range(len(decisions)):
        if (allow_repetition[i] == False) and (decisions[i] == decision_history[i]):
            return True
    return False


# this class is a fake LLM interface that accepts the prompt from the LLM and gives it to us
# it has the same structure as the real LLM interfaces
class fake_model:
    def __init__(self, model):
        self.model = model

    def __call__(self, messages, temperature=0.0):
        # put it all in the user prompt
        self.latest_prompt = [{"role": "user", "content": messages[0]['content'] + messages[1]['content']}]
        return "<Answer>None</Answer>"

    def return_prompt(self):
        return self.latest_prompt


# history_file is a pandas pickle from which we will get the history
# solver_name and replicate_number help us label the solver priving the history as well
# a is the index to start, b is the index to end
def generate_prompts(history_file, solver_name, replicate_number, a, b, decision_history):

    # read in the history dataframe
    hist = pd.read_pickle(history_file)

    # sanity check
    if hist.shape[0] < b:
        raise Exception(f"b = {b}, hist.shape[0] = {hist.shape[0]}. So b is greater than hist.shape[0]")


    num_arms = 5        # this number cannot be changed without significant changes to the code right now
    arms = np.array(['blue','green','purple','red','yellow'])
    arm_ids = dict(zip(arms, range(num_arms)))

    # make our solvers (they will give us their opinion on the next color)
    solvers = []
    for i in solver_configurations:
        if i['type'] == 'Greedy':
            solvers.append(Greedy(i['name'], i['epsilon'], initial_action_values=i['initial_action_values'], try_each_once=i['try_each_once']))
        elif i['type'] == 'UCB':
            solvers.append(UCB(i['name'], const=i['const'], initial_values=i['initial_values'], log_base=i['log_base']))
        elif i['type'] == 'TS':
            solvers.append(TS(i['name']))
        elif i['type'] == 'Random':
            solvers.append(Random(i['name']))


    # call setup() on each solver
    for solver in solvers:
        solver.setup("blue, green, red, yellow, purple", arms, 100, True)

    # make the LLM that will generate the prompts
    model = fake_model('fake')
    llm = LLM(LLM_config['name'], model=model, balance_EE=LLM_config['balance_EE'], end_with_goal=LLM_config['end_with_goal'], temperature=LLM_config['temperature'], short_answers=LLM_config['short_answers'], save_every_nth_response=LLM_config['save_every_nth_response'], provide_context_on_round=LLM_config['provide_context_on_round'])

    # setup the llm
    llm.setup("blue, green, red, yellow, purple", arms, 100, True)


    # ok, now we go through the first a steps of the history and append them to our solvers and LLM
    for i in range(a):
        chosen_arm = hist.loc[i, 'chosen_arm']
        reward = hist.loc[i, 'reward']
        index_of_chosen_arm = arm_ids[chosen_arm]

        # update our solvers
        for solver in solvers:
            solver.update(chosen_arm, index_of_chosen_arm, reward)

        # update the llm we will use to generate the prompt
        llm.update(chosen_arm, index_of_chosen_arm, reward)


    # alright, now we are ready to generate actual prompts

    # number of trials to do
    num_trials = b - a
    # line we are up to
    line = a

    for i in range(num_trials):

        # get the actual facts
        verdict = hist.loc[line, 'chosen_arm']
        verdict_index = arm_ids[verdict]
        recieved_reward = hist.loc[line, 'reward']

        decisions = []
        # get the decisions of the solvers i.e. what they think the facts should be :)
        for solver in solvers:
            chosen_arm, _ = solver.make_decision()
            decisions.append(chosen_arm)
            # update each one while we're at it
            solver.update(verdict, verdict_index, recieved_reward)

        # append the verdict of history itself!
        decisions.append(verdict)


        # check for agreement
        if too_much_agreement(decisions, agreement) or too_much_repetition(decisions, allow_repetition, decision_history):
            llm.update(verdict, verdict_index, recieved_reward)
            line+=1
            continue

        # get the prompt from the LLM
        llm.make_decision()
        prompt = model.return_prompt()

        prompt.append({'role': 'assistant', 'content': ''})

        for arm in arms:

            # solvers that endorse this arm on this trial
            endorsements = []
            for i in range(len(solvers)):
                if decisions[i] == arm:
                    endorsements.append(solvers[i].name)

            if verdict == arm:
                endorsements.append(solver_name)

            if len(endorsements) > 0:
                prompt[-1]['content'] = f"<Answer>{arm}</Answer>"


                # the solvers who gave the answer expressed in this file
                endorsers = ""
                for endorser in endorsements:
                    endorsers += ', ' + endorser

                filename = solver_name + ' #' + str(replicate_number) + ' step ' + str(line) + endorsers
                filepath = os.path.join(output_directory, filename)
                with open(filepath, "w") as f:
                    json.dump(prompt, f)

        llm.update(verdict, verdict_index, recieved_reward)
        decision_history = decisions.copy()
        line+=1


if __name__ == "__main__":

    decision_history = [None]*(len(solver_configurations)+1)

    history_pickles = ["llama3 temp=0.01"]

    for pickle in history_pickles:
        for replicate_number in range(24):
            generate_prompts(os.path.join(input_directory, pickle + ' #' + str(replicate_number)), pickle, replicate_number, 20, 85, decision_history)

