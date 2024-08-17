from nnsight import LanguageModel
import torch
import numpy as np

class llama3_record_activations:
    # file_name_pattern will form the basis of filenames which will all be concatenated with
    # numbers like so: file_name_pattern_i.npy
    def __init__(self, name, hf_key, token_positions_to_extract, file_name_pattern):

        self.model = name
        torch.autograd.set_grad_enabled(False)

        self.BLOCKS = 32
        self.token_positions_to_extract = token_positions_to_extract
        self.file_name_pattern = file_name_pattern
        self.file_num = 0
        self.number = 0

        self.actual_model = LanguageModel(name, device_map="auto", token=hf_key)


    def __call__(self, messages, temperature=0.0, max_tokens=50):

        # assign it all to the user
        prompt = [{"role": "user", "content": messages[0]['content'] + messages[1]['content']}]

        input_ids = self.actual_model.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # for each block, we want the output of the block as a whole
        with self.actual_model.trace(input_ids):
            activations = [
                self.actual_model.model.layers[i].output[0].save()
                for i in range(self.BLOCKS)
            ]

        # we now have a list of length blocks, where each element is (1, num_tokens, dimensions)

        for i in range(self.BLOCKS):
            # make the ndarray (num_tokens, dimensions)
            activations[i] = np.swapaxes(activations[i].cpu().numpy(), 0, 1).squeeze()
            # make a new array
            acts = []
            # cut out all tokens except the ones we want
            for j in range(len(self.token_positions_to_extract)):
                acts.append(activations[i][self.token_positions_to_extract[j]])

            # make activations[i] into array (num_tokens_that_we_want, dimensions)
            activations[i] = np.stack(acts)

        # now we get (num_layers, num_tokens_that_we_want, dimensions)
        activations = np.stack(activations)

        np.save(self.file_name_pattern + '_' + str(self.number) + '.npy', activations)
        self.number+=1

        return "<Answer>No Answer</Answer>"


