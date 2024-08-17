import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from statistics import fmean, median
import os

# retrieves a dataset for the models specified in the list averaged over num_replicates
def get_data(data_folder, models, num_replicates, stat):

    data = dict()

    for model in models:

        columns = []

        for i in range(num_replicates):
            filename = f"{model} #{i}"
            filepath = os.path.join(data_folder, filename)
            columns.append(pd.read_pickle(filepath)[stat])

        horizontal_concatenation = pd.concat(columns, axis=1)

        data[model] = horizontal_concatenation.mean(axis=1)

    return data



def num_stat(data_folder, models, stat, num_replicates, num_steps):
    ''' plot a specific stat like k*min_frac or average reward
        over all time steps averaged over all replicates
    '''

    data = get_data(data_folder, models, num_replicates, stat)

    for model in models:
        x = np.arange(num_steps)
        y = data[model] # y could be k*minfrac or something else
        plt.plot(x, y, label=model)

    plt.xlabel("Steps")
    plt.ylabel(stat)
    plt.title(f"{stat} for each model averaged over {num_replicates} replicates")
    plt.legend()
    plt.grid(True)

    plt.show()
    

def last_step_average_reward_by_replicate(models, data_folder, num_replicates, last_step):
    ''' plots the final avergae reward (at final step) over different replicates for each model
    
        with more time steps with this function we can differentiate the final average rewards each model got
        and how consistent they are with these values
    '''
        
    last_step -= 1
    for model in models:
        avg_rewards = []  
        
        for i in range(num_replicates):
            filename = f"{model} #{i}" 
            filepath = os.path.join(data_folder, filename)
            data = pd.read_pickle(filepath)
            
            # Extract average reward for the last step of the current simulation
            avg_reward = data[data['step'] == last_step]['average reward'].values[0]
            avg_rewards.append(avg_reward)
        
        
        plt.plot(range(num_replicates), avg_rewards, label=model)

    plt.xlabel("Simulations")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Simulation (Last Step of Each Simulation for Each Model)")
    plt.legend()
    plt.grid(True)

    plt.show()
    


def last_step_average_reward(models, data_folder, num_replicates, last_step):

    # this will give us the average, average rewards for each model on each step
    # that's more than we need. We will cut out the last row
    data = get_data(data_folder, models, num_replicates, "average reward")

    for model in models:
        data[model] = data[model].iat[last_step-1]

    models = list(data.keys())
    average_rewards = list(data.values())

    fig = plt.figure(figsize=(20,15))
    plt.bar(models, average_rewards)
    plt.show()




def simulation_heatmap(models, data_folder, simulation_num):
    ''' heat map for all the models comparing k*min_frac and average reward
        like the krishnamurthi et al paper
    '''
    avg_rewards = []
    k_minfracs = []
    
    for model in models:
        filename = f"{model} #{simulation_num}" 
        filepath = os.path.join(data_folder, filename)
        data = pd.read_pickle(filepath)
        
        avg_reward = data[data['step'] == 99]['average reward'].values[0]
        k_minfrac = data[data['step'] == 99]['k*min_frac'].values[0]
        
        avg_rewards.append(avg_reward)
        k_minfracs.append(k_minfrac)

    # Created df for heatmap
    df = pd.DataFrame({'average Reward': avg_rewards, 'k*min_frac': k_minfracs}, index=models)

    print(df)

    sns.heatmap(df, annot=True, cmap = "crest") #annot writes the numbers on the squres and cmap descides color
    plt.title(f"Heatmap of Average Reward and k*minfrac for Simulation {simulation_num}")
    plt.xlabel("Metrics")
    plt.ylabel("Models")
    plt.show()


def krishnamurthi_heatmap(many_replicates, few_replicates, many_replicate_models, few_replicate_models, many_replicate_folder, few_replicate_folder):

    # get from JSON: MedianReward, SuffFailFreq(T/2), GreedyFrac
    # get from frames: K*Minfrac()

    many_json_file_data = json_data(many_replicate_folder)
    few_json_file_data = json_data(few_replicate_folder)

    many_reward_data = many_json_file_data.get_summarized_data(many_replicate_models, "Total reward:", many_replicates, average=False)
    few_reward_data = few_json_file_data.get_summarized_data(few_replicate_models, "Total reward:", few_replicates, average=False)

    reward_data = []

    # normalize and append
    for model in many_replicate_models:
        reward_data.append(many_reward_data[model]/100.0)
    for model in few_replicate_models:
        reward_data.append(few_reward_data[model]/100.0)

    many_suffix_data = many_json_file_data.get_summarized_data(many_replicate_models, "Suffix Fail (T/2):", many_replicates, average=True)
    few_suffix_data = few_json_file_data.get_summarized_data(few_replicate_models, "Suffix Fail (T/2):", few_replicates, average=True)

    suffix_data = []

    # append
    for model in many_replicate_models:
        suffix_data.append(many_suffix_data[model])
    for model in few_replicate_models:
        suffix_data.append(few_suffix_data[model])

    many_greedy_data = many_json_file_data.get_summarized_data(many_replicate_models, "Greedy Frac:", many_replicates, average=True)
    few_greedy_data = few_json_file_data.get_summarized_data(few_replicate_models, "Greedy Frac:", few_replicates, average=True)

    greedy_data = []

    # append
    for model in many_replicate_models:
        greedy_data.append(many_greedy_data[model])
    for model in few_replicate_models:
        greedy_data.append(few_greedy_data[model])


    num_replicates = []

    # append
    for model in many_replicate_models:
        num_replicates.append(many_replicates)
    for model in few_replicate_models:
        num_replicates.append(few_replicates)


    many_k_min_frac = get_data(many_replicate_folder, many_replicate_models, many_replicates, "k*min_frac")
    few_k_min_frac = get_data(few_replicate_folder, few_replicate_models, few_replicates, "k*min_frac")

    for model in many_replicate_models:
        many_k_min_frac[model] = many_k_min_frac[model].iat[99]
    for model in few_replicate_models:
        few_k_min_frac[model] = few_k_min_frac[model].iat[99]

    k_min_frac_data = []

    # append
    for model in many_replicate_models:
        k_min_frac_data.append(many_k_min_frac[model])
    for model in few_replicate_models:
        k_min_frac_data.append(few_k_min_frac[model])


    table = [reward_data, suffix_data, k_min_frac_data, greedy_data]



    # k_minfracs = []
    #
    # for model in models:
    #     filename = f"{model} #{simulation_num}"
    #     filepath = os.path.join(data_folder, filename)
    #     data = pd.read_pickle(filepath)
    #
    #     avg_reward = data[data['step'] == 99]['average reward'].values[0]
    #     k_minfrac = data[data['step'] == 99]['k*min_frac'].values[0]
    #
    #     avg_rewards.append(avg_reward)
    #     k_minfracs.append(k_minfrac)

    # Created df for heatmap
    df = pd.DataFrame(table, index=["MedianReward", "SuffFailFreq(T/2)", "K*MinFrac", "GreedyFrac"], columns=many_replicate_models+few_replicate_models)

    print(df)

    fig = sns.heatmap(df, annot=True, cmap = "crest", fmt='g') #annot writes the numbers on the squres and cmap descides color
    plt.tick_params(labelbottom = False, labeltop = True)
    plt.show()


# I'm making this a class so we only need to read the json file once
# even if we repeatedly issue calls for different summarized stats
class json_data:
    def __init__(self, data_folder):
        filepath = os.path.join(data_folder, "output.json")

        self.raw_data = dict()

        with open(filepath) as f:
            for line in f.readlines():
                d = json.loads(line)
                self.raw_data[d["Solver name:"]] = d

    # get summarized data for a stat
    # if average is false, we take the median instead
    def get_summarized_data(self, models, stat, num_replicates, average=True):
        summarized_data = dict()

        # init
        for model in models:
            summarized_data[model] = list()

        for model in models:
            for i in range(num_replicates):
                summarized_data[model].append(self.raw_data[f"{model} #{i}"][stat])

        if average:
            for model in models:
                summarized_data[model] = fmean(summarized_data[model])
        else:
            for model in models:
                summarized_data[model] = median(summarized_data[model])

        return summarized_data



# eventually delete this deprecated function and replace with instantiations of the class above
def read_json_data(models, data_folder, stat, num_replicates, average=True):

    filepath = os.path.join(data_folder, "output.json")

    raw_data = []

    with open(filepath) as f:
        for line in f.readlines():
            raw_data.append(json.loads(line))

    summarized_data = dict()

    # init
    for model in models:
        summarized_data[model] = list()

    for model in models:
        for i in range(num_replicates):
            for row in raw_data:
                if row['Solver name:'] == f"{model} #{i}":
                    summarized_data[model].append(row[stat])

    if average:
        for model in models:
            summarized_data[model] = fmean(summarized_data[model])
    else:
        for model in models:
            summarized_data[model] = median(summarized_data[model])

    return summarized_data


def bar_plot_mean_suffix_point(models, data_folder, num_replicates):

    data = read_json_data(models, data_folder, "Suffix point:", num_replicates)

    models = list(data.keys())
    suffix_points = list(data.values())

    fig = plt.figure(figsize=(20,15))
    plt.bar(models, suffix_points)
    plt.show()


def bar_plot_suffix_failure_rate(models, data_folder, num_replicates):

    data = read_json_data(models, data_folder, "Suffix Fail (T/2):", num_replicates)

    models = list(data.keys())
    failure_rate = list(data.values())

    fig = plt.figure(figsize=(20,15))
    plt.bar(models, failure_rate)
    plt.show()



# models = ["Random", "Greedy e=0", "Greedy e=0.1", "Greedy e=0.01", "UCB", "Thompson Sampling","gpt-3.5-turbo temp=1.0","gpt-3.5-turbo temp=0.0"]
# models = ["Greedy e=0", "Greedy e=0.1", "Greedy e=0.01", "UCB", "Thompson Sampling"]

many_replicate_models = ["Thompson Sampling", "UCB", "Greedy e=0", "Greedy e=0.1", "Greedy e=0.01"]
few_replicate_models = ["llama3 temp=1.0","llama3 temp=0.01"]
many_replicates = 30
few_replicates = 30
many_replicate_folder = "results"
few_replicate_folder = "results"
krishnamurthi_heatmap(many_replicates, few_replicates, many_replicate_models, few_replicate_models, many_replicate_folder, few_replicate_folder)

# data_folder = "Thompson-Greedy-UCB"
# data_folder = "attempt2"
# replicates = 30
# steps = 100
# stat = "k*min_frac"
# stat = "average reward"

# simulation_heatmap(models, data_folder, 29)

#so far only shows final average reward
# last_step_average_reward_by_replicate(models, data_folder, replicates, steps)
# last_step_average_reward(models, data_folder, replicates, steps)

# bar_plot_suffix_failure_rate(models, data_folder, replicates)

#can show any stat - k*min_frac, average reward, for listed models averaged over a number of replicates
# num_stat(data_folder, models, stat, replicates, steps)
