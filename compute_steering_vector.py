import numpy as np
import os

layer = 13
prompts = 400
dimensions = 4096      # this matters only for the starting vector

# we put two numbers here corresponding to the prompts
# numbers at which these solvers begin
# i.e. we will compute the steering vector over
# promtps beginning at these numbers and going for <prompts> prompts
solvers = (0, 400)
directory = "activations_4"

if __name__ == "__main__":

    os.chdir(directory)

    begin_plus = solvers[0]
    end_plus = solvers[0]+prompts
    begin_minus = solvers[1]
    end_minus = solvers[1]+prompts

    n=1
    steering_vec = np.zeros(dimensions)

    for i, j in zip(range(begin_plus, end_plus), range(begin_minus, end_minus)):
        arr_plus = np.load(f"prompt_{i}_layer_{layer}.npy").squeeze()
        arr_minus = np.load(f"prompt_{j}_layer_{layer}.npy").squeeze()

        diff = arr_plus - arr_minus

        # update the running average
        steering_vec += 1/n*(diff - steering_vec)

        n+=1

    # save the steering vector
    os.chdir("..")
    np.save("steering_vector.npy", steering_vec)
