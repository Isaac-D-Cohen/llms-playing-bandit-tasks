from lib.MAB import MAB
from lib.Solvers.LLM import LLM
from lib.Solvers.Greedy import Greedy
from lib.Solvers.UCB import UCB
from lib.Solvers.TS import TS
from lib.Solvers.Random import Random
from lib.Solvers.LLM_interfaces.llama3_8b import llama3
from tqdm import tqdm
import json
import traceback

from sys import argv, exit


if __name__ == "__main__":

    if len(argv) < 2:
        print(f"Usage: {argv[0]} <hf_key>")
        exit(0)

    hf_key = argv[1]

    T = 100
    bernoulli = True


    # create the MAB instance
    bandit_instance = MAB(T=T, bernoulli=bernoulli)

    # create the solvers
    solvers = []
    solvers.append(Greedy(f"Greedy e=0"))
    solvers.append(Greedy(f"Greedy e=0.001", 0.001, initial_action_values=0.0))
    solvers.append(Greedy(f"Greedy e=0.01", 0.01, initial_action_values=0.0))
    solvers.append(Greedy(f"Greedy e=0.1", 0.1, initial_action_values=0.0))
    solvers.append(Random(f"Random"))
    solvers.append(UCB(f"UCB", log_base=10 ,initial_values=0))
    solvers.append(TS(f"Thompson Sampling"))
    model = llama3("meta-llama/Meta-Llama-3-8B-Instruct", hf_key)
    solvers.append(LLM(f"llama3 temp=0.01", model=model, balance_EE=True, end_with_goal=True, temperature=0.01, save_every_nth_response=10, short_answers=True))
    model = llama3("meta-llama/Meta-Llama-3-8B-Instruct", hf_key)
    solvers.append(LLM(f"llama3 temp=1.0", model=model, balance_EE=True, end_with_goal=True, temperature=1.0, save_every_nth_response=10, short_answers=True))


    # record everything
    with open("metadata.json", "w") as f:
        # record MAB
        json.dump(bandit_instance.get_params(), f)
        f.write("\n\n")

        # record solvers
        for solver in solvers:
            json.dump(solver.get_params(), f)
            f.write("\n")


    results = []

    try:
        for i in range(30):

            print(f"On iteration {i}...")

            solvers[0].change_name(f"Greedy e=0 #{i}")
            solvers[1].change_name(f"Greedy e=0.001 #{i}")
            solvers[2].change_name(f"Greedy e=0.01 #{i}")
            solvers[3].change_name(f"Greedy e=0.1 #{i}")
            solvers[4].change_name(f"Random #{i}")
            solvers[5].change_name(f"UCB #{i}")
            solvers[6].change_name(f"Thompson Sampling #{i}")
            solvers[7].change_name(f"llama3 temp=0.01 #{i}")
            solvers[8].change_name(f"llama3 temp=1.0 #{i}")

            bandit_instance.fresh_instance()
            results.append(bandit_instance.simulate(solvers))
    except Exception:
        print(f"An exception occured. Exiting early\n{traceback.format_exc()}")

    finally:
        with open("output.json", "w") as f:
            for i in tqdm(range(len(results))):
                for result in results[i]:
                    json.dump(result.get_summary(T), f)
                    f.write("\n")
                    result.pickle_it(result.solver_name)
