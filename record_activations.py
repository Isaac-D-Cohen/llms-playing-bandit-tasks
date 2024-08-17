import json
import os
import numpy as np
from sys import argv
from nnsight import LanguageModel
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

input_directory = "processed_outputs"

solvers = ["RealGreedy e=0", "AntiGreedy e=0"]         # which solvers we have (we can name greedy prompts 'RealGreedy' so they don't get confused with AntiGreedy)
num_prompts = 400                                      # how many of each we have

BLOCKS = 32

def get_activations(inputs):

    # for each block, we want the output of the block as a whole
    with model.trace(inputs):
        activations = [
            model.model.layers[i].output[0].save()
            for i in range(BLOCKS)
        ]

    # we now have a list of length blocks, where each element is (1, num_tokens, dimensions)

    for i in range(BLOCKS):
        # for make the ndarray (num_tokens, 1, dimensions)
        activations[i] = np.swapaxes(activations[i].cpu().numpy(), 0, 1)
        # cut out all tokens but the ninth last (corresponding to the color chosen)
        activations[i] = activations[i][-9]
        # so now we have (1, dimensions)

    return activations

if __name__ == "__main__":

    if len(argv) < 2:
        print(f"Usage: {argv[0]} <hf_key>")
        exit(0)

    hf_key = argv[1]

    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    model = LanguageModel("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", token=hf_key)

    prompts = []

    for solver in solvers:
        print(f"Now reading solver {solver}")
        for i in tqdm(range(num_prompts)):
            filename = os.path.join(input_directory, solver + ' ' + str(i) + '.json')
            with open(filename, "r") as f:
                current_prompt = json.load(f)

            input_ids = tokenizer.apply_chat_template(
                current_prompt,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            prompts.append(input_ids)

    print("Ok, now running prompts...")

    for j in tqdm(range(len(prompts))):
        activations = get_activations(prompts[j])
        for i in range(BLOCKS):
            np.save(f"prompt_{j}_layer_{i}", activations[i])
