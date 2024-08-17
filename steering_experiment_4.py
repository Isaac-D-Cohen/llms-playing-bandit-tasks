from lib.MAB import MAB
from lib.Solvers.LLM import LLM
from lib.Solvers.Greedy import Greedy
from lib.Solvers.AntiGreedy import AntiGreedy
from lib.Solvers.Random import Random
from lib.Solvers.LLM_interfaces.llama3_8b_with_steering import steered_llama3
from tqdm import tqdm
import json
import torch
import numpy as np
import traceback
from sys import argv

if __name__ == "__main__":

    if len(argv) < 2:
        print(f"Usage: {argv[0]} <hf_key>")
        exit(0)

    hf_key = argv[1]

    n = 15      # 15 times per bandit instance
    k = 20      # 20th trial

    # for the bandit instance
    T = 100
    bernoulli = True

    # read in the steering vector
    vec = torch.from_numpy(np.load("greedy_vector.npy"))

    # create the MAB instance
    bandit_instance = MAB(T=T, bernoulli=bernoulli)

    # create the history_makers
    history_makers = []
    history_makers.append(Random(f"Random"))
    history_makers.append(Random(f"Random"))
#    history_makers.append(Random(f"Random"))

    # create the deciders
    deciders = []

    model = steered_llama3("meta-llama/Meta-Llama-3-8B-Instruct", hf_key, steering_vec=2*vec, layer_num=13)
    deciders.append(LLM(f"llama3 positive", model=model, balance_EE=True, end_with_goal=True, temperature=0.01, save_every_nth_response=10, short_answers=True))
    model = steered_llama3("meta-llama/Meta-Llama-3-8B-Instruct", hf_key, steering_vec=(-2)*vec, layer_num=13)
    deciders.append(LLM(f"llama3 negative", model=model, balance_EE=True, end_with_goal=True, temperature=0.01, save_every_nth_response=10, short_answers=True))

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



    grand_summary_array = []
    for decider in deciders:
        grand_summary_array.append({'Name': decider.name + ' - All', 'Greedy': 0.0, 'Optimal': 0.0, 'Invalid': 0})

    f = open("output.json", "w")

    try:
        # this loop starts at 1 so we don't divide by 0 when we do 1/i
        for i in range(1, 31):
            print(f"On iteration {i}...")

            bandit_instance.fresh_instance()
            stat_trackers, summary_array, dataframe_array = bandit_instance.run_alternate_simulation(deciders, history_makers, n, k, i)

            # write out the pickles
            for tracker in stat_trackers:
                tracker.pickle_it(tracker.solver_name)

            for j in range(len(dataframe_array)):
                dataframe_array[j].to_pickle(deciders[j].name + ' #' + str(i))


            for j in range(len(deciders)):
                grand_summary_array[j]['Greedy'] += 1/i * (summary_array[j]['Greedy'] - grand_summary_array[j]['Greedy'])
                grand_summary_array[j]['Optimal'] += 1/i * (summary_array[j]['Optimal'] - grand_summary_array[j]['Optimal'])
                grand_summary_array[j]['Invalid'] += summary_array[j]['Invalid']

                json.dump(summary_array[j], f)
                f.write('\n')


    except Exception:
        print(f"An exception occured. Exiting early\n{traceback.format_exc()}")

    finally:
        for summary in grand_summary_array:
            json.dump(summary, f)
            f.write('\n')
        f.close()


