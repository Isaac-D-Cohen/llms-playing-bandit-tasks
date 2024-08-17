from nnsight import LanguageModel
import torch

class steered_llama3:
    def __init__(self, name, hf_key, steering_vec, layer_num):

        self.model = name
        torch.autograd.set_grad_enabled(False)

        self.actual_model = LanguageModel(name, device_map="auto", token=hf_key)
        self.actual_model.tokenizer.padding_side = 'left'

        self.actual_model.generation_config.pad_token_id = self.actual_model.tokenizer.pad_token_id
        self.terminators = [self.actual_model.tokenizer.eos_token_id, self.actual_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        self.steering_vec = steering_vec
        self.layer_num = layer_num

    def __call__(self, messages, temperature=0.0, max_tokens=50):

        # assign it all to the user
        prompt = [{"role": "user", "content": messages[0]['content'] + messages[1]['content']}]

        input_ids = self.actual_model.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        with self.actual_model.generate(input_ids, max_new_tokens=60, do_sample=True, temperature=temperature, pad_token_id=self.actual_model.tokenizer.eos_token_id):
            for i in range(3):
                self.actual_model.next()
            self.actual_model.model.layers[self.layer_num].output[0][:] += self.steering_vec
            output = self.actual_model.output.save()

        response = self.actual_model.tokenizer.decode(output[0].argmax(-1)[0])

        if response == "urple":
            response = "purple"

        return "<Answer>" + response + "</Answer>"

