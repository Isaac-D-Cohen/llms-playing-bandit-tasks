from openai import OpenAI 

class GPT:

    def __init__(self, model, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def __call__(self, messages, temperature=0.0):
        response = self.client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = temperature
        )
        return response.choices[0].message.content
