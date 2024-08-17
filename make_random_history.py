from lib.MAB import MAB
from lib.Solvers.Random import Random
from tqdm import tqdm
import json
import os

from sys import argv, exit

output_directory = "inputs"

if __name__ == "__main__":

    T = 10
    bernoulli = True

    # create the MAB instance
    bandit_instance = MAB(T=T, bernoulli=bernoulli)

    # create the solvers
    solvers = [Random(f"Random")]

    # record everything
    with open(os.path.join(output_directory, "metadata.json"), "w") as f:
        # record MAB
        json.dump(bandit_instance.get_params(), f)
        f.write("\n\n")

        # record solvers
        for solver in solvers:
            json.dump(solver.get_params(), f)
            f.write("\n")


    results = []

    for i in range(575):

        print(f"On iteration {i}...")

        solvers[0].change_name(f"Random #{i}")

        bandit_instance.fresh_instance()
        results.append(bandit_instance.simulate(solvers))


    with open(os.path.join(output_directory, "output.json"), "w") as f:
        for i in tqdm(range(len(results))):
            for result in results[i]:
                json.dump(result.get_summary(T), f)
                f.write("\n")
                result.pickle_it(os.path.join(output_directory, result.solver_name))
