from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class llama3:
    def __init__(self, name, hf_key):

        self.model = name
        torch.autograd.set_grad_enabled(False)

        self.actual_model = AutoModelForCausalLM.from_pretrained(
            name,
            token=hf_key,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(name, token=hf_key)
        self.actual_model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    def __call__(self, messages, temperature=0.0, max_tokens=50):

        # assign it all to the user
        prompt = [{"role": "user", "content": messages[0]['content'] + messages[1]['content']}]

        input_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.actual_model.device)

        outputs = self.actual_model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = outputs[0][input_ids.shape[-1]:]

        return self.tokenizer.decode(response, skip_special_tokens=True)

