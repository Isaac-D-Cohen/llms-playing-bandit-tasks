import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import os
import re

prompt_directory = "processed_outputs"
activations_directory = "activations_4"
solvers = ["RealGreedy e=0", "AntiGreedy e=0"]         # which solvers we have
num_prompts = 400                                  # how many prompts we have of each
num_layers = 32                                     # how many layers the model has


def make_pca_graph(activations, answers_array, labels, layer):
    # do the PCA
    pca = PCA(2)
    x = StandardScaler().fit_transform(activations)
    principalComponents = pca.fit_transform(x)


    print(f"Layer {layer}: {pca.explained_variance_ratio_}")

    # make the dataframe
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

    labels = np.stack((answers_array, labels), axis=1)
    finalDf = pd.concat([principalDf, pd.DataFrame(data = labels, columns = ["answers", "labels"])], axis = 1)

    # make the graph
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    markers = ["o", "^"]
    targets = ["Greedy e=0", "AntiGreedy e=0"]
    color_names = ['blue', 'green', 'red', 'yellow', 'purple']
    colors_rgb = [(0.0, 0.15294117647058825, 1.0), (0.0, 0.8431372549019608, 0.0), (0.9294117647058824, 0.12156862745098039, 0.0),
                    (0.9294117647058824, 0.9803921568627451, 0.0), (0.5294117647058824, 0.1607843137254902, 1.0)]

    for target, marker in zip(targets, markers):
        for color, rgb_val in zip(color_names, colors_rgb):
            indicesToKeep = finalDf[(finalDf['labels'] == target) & (finalDf['answers'] == color)]
            if len(indicesToKeep) > 0:
                handle = ax.scatter(indicesToKeep['principal component 1']
                        , indicesToKeep['principal component 2']
                        , color = rgb_val
                        , marker = marker
                        , s = 50)
    ax.grid()
    plt.savefig(f"results_layer_{layer}.png", dpi = 500)
    plt.close()



def do_pca(activations, n_components):
    # do the PCA
    pca = PCA(n_components)
    x = StandardScaler().fit_transform(activations)
    principalComponents = pca.fit_transform(x)
    return principalComponents



def main1():

    labels = ["Greedy e=0"]*num_prompts + ["AntiGreedy e=0"]*num_prompts

    for i in range(num_layers):

        activations_array = []
        answers_array = []

        # absolute prompt number
        n=0

        for solver in solvers:
            for j in range(num_prompts):
                activations_path = os.path.join(activations_directory, f"prompt_{n}_layer_{i}.npy")
                act = np.load(activations_path).squeeze()   # we cut out the first dimension immediately (because it's just 1)
                activations_array.append(act)

                # get the color from the prompt
                prompt_filename = os.path.join(prompt_directory, f"{solver} {j}.json")
                with open(prompt_filename, "r") as f:
                    prompt = json.load(f)
                    answer = prompt[1]['content']

                    # make a regex to extract the answer
                    pattern = r"<Answer>(.*?)</Answer>"
                    matches = re.findall(pattern, answer)
                    color = matches[0].strip().lower()
                    answers_array.append(color)

                n+=1

        activations = np.stack(activations_array)
        make_pca_graph(activations, answers_array, labels, i)



def main2():

    pc_values = []

    for i in range(num_layers):

        activations_array = []
        answers_array = []

        # absolute prompt number
        n=0

        for solver in solvers:
            for j in range(num_prompts):
                activations_path = os.path.join(activations_directory ,f"prompt_{n}_layer_{i}.npy")
                act = np.load(activations_path).squeeze()   # we cut out the first dimension immediately (because it's just 1)
                activations_array.append(act)
                n+=1

        activations = np.stack(activations_array)
        pc_values.append(do_pca(activations, 6))


    # saving pc_values
    pc_values = np.array(pc_values) # (32,52,5)
    np.save(f'pc_values', pc_values)


if __name__ == "__main__":

    # main1()
    main2()
