import requests
import time

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

class Mixtral:

    def __init__(self, name, hf_key):
        self.model = name
        self.headers = {"Authorization": f"Bearer {hf_key}"}

    def __call__(self, messages, temperature=0.0):
        prompt = f"<s> [INST] {messages[0]['content'] + messages[1]['content']} [/INST]"
        length = len(prompt)
        payload = {
            "inputs": prompt,
            "parameters": {"temperature": temperature, "max_length": 400},
        }

        attempts=0
        max_retries=15

        while attempts < max_retries:
            response = requests.post(API_URL, headers=self.headers, json=payload)

            if response.status_code == 200:
                break
            else:
                print(response)
                print(response.headers)

            attempts += 1
            time.sleep(0.25)

        if attempts == max_retries:
            raise Exception (f"Couldn't get answer from LLM after {attempts} tries")

        result = response.json()[0]['generated_text']
        return result[length:]
