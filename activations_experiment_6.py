from lib.MAB import MAB
from lib.Solvers.LLM import LLM
from lib.Solvers.UCB import UCB
from lib.Solvers.LLM_interfaces.llama38b_record_activations import llama3_record_activations
from tqdm import tqdm
import json
import traceback
import os

from sys import argv, exit

output_dir = "outputs"

if __name__ == "__main__":

    os.chdir(output_dir)

    if len(argv) < 2:
        print(f"Usage: {argv[0]} <hf_key>")
        exit(0)

    hf_key = argv[1]

    T = 100
    k = 40
    bernoulli = True


    # create the MAB instance
    bandit_instance = MAB(T=T, bernoulli=bernoulli)

    # create the history_makers
    history_makers = []
    history_makers.append(UCB(f"UCB", log_base=10 ,initial_values=0))

    # create the deciders
    deciders = []
    # figure out what goes into this array
    token_positions_to_extract = [-21, -19, -17, -15, -13]
    model = llama3_record_activations("meta-llama/Meta-Llama-3-8B-Instruct", hf_key, token_positions_to_extract, "activations")
    deciders.append(LLM(f"llama3 temp=0.01", model=model, balance_EE=True, end_with_goal=True, temperature=0.01, save_every_nth_response=10, short_answers=True))


    # record everything
    with open("metadata.json", "w") as f:
        # record MAB
        json.dump(bandit_instance.get_params(), f)
        f.write("\n\n")

        # the first half will always be the history_makers
        # the second half, the deciders
        # record history_makers
        for solver in history_makers:
            json.dump(solver.get_params(), f)
            f.write("\n")

        # record deciders
        for solver in deciders:
            json.dump(solver.get_params(), f)
            f.write("\n")


    for replicate_number in range(100):
        print(f"On iteration {replicate_number}...")

        bandit_instance.fresh_instance()
        stat_trackers, ucb_value_frames = bandit_instance.run_alternate_simulation2(deciders, history_makers, k, replicate_number)

        # write out the pickles
        for tracker in stat_trackers:
            tracker.pickle_it(tracker.solver_name)

        for j in range(len(ucb_value_frames)):
            ucb_value_frames[j].to_pickle(deciders[j].name + ' #' + str(replicate_number))

