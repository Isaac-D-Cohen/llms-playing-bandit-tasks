import json
import os
from shutil import copyfile
import numpy as np
import re

input_directory = "outputs"
output_directory = "processed_outputs"


# technically there can be an issue where a solver name
# contains another solver name within it, and is confused for being both
# this code doesn't address that issue
solvers = ["RealGreedy e=0", "AntiGreedy e=0"]
solver_count = [0 for _ in range(len(solvers))]

# a class to track how often a solver uses a certain color
class stats:
    def __init__(self, name):
        self.solver_name = name
        self.colors = {"blue": 0, "red": 0, "green": 0, "purple": 0, "yellow": 0}

    def update_color(self, color):
        self.colors[color] += 1

    def get_colors(self):
        return (self.solver_name, self.colors)

if __name__ == "__main__":

    # array that is length of solvers to keep track the colors stats
    color_stats = [stats(name) for name in solvers]

    prompt_files = os.listdir(input_directory)

    for prompt_file in prompt_files:

        name = prompt_file.split('#')[1]

        for i in range(len(solvers)):

            if solvers[i] in name:
                # copy the file and give it a name in the format <solver>_<i>.json ex. Greedy e=0 5.json
                source = os.path.join(input_directory, prompt_file)
                destination = os.path.join(output_directory, solvers[i] + ' ' + str(solver_count[i]) + '.json')
                copyfile(source, destination)

                # add 1 to the count for this solver
                solver_count[i] += 1

                # and 1 to the count for the color in this prompt
                with open(destination, "r") as f:
                    p = json.load(f)
                    answer = p[1]['content']

                    # make a regex to extract the answer
                    pattern = r"<Answer>(.*?)</Answer>"
                    matches = re.findall(pattern, answer)
                    color = matches[0].strip().lower()

                    color_stats[i].update_color(color)

    for s in color_stats:
        print(s.get_colors())

