import re
import json
import traceback

class LLM:
    def __init__(self, name, model, balance_EE=False, short_answers=False, end_with_goal=False, temperature=0.0, save_every_nth_response=0, provide_context_on_round=False):
        self.name = name
        self.model = model
        self.balance_EE = balance_EE
        self.short_answers = short_answers
        self.temperature = temperature
        self.end_with_goal = end_with_goal
        self.provide_context_on_round = provide_context_on_round

        # setting this to 0 means save no responses, 1 means every response, any other number means every nth response
        self.save_every_nth_response = save_every_nth_response

    def change_name(self, name):
        self.name = name

    def setup(self, labels, arms, T=100, bernoulli=True):

        self.labels = labels
        self.history = ""
        self.step = 0
        self.bernoulli_or_gauss = "Bernoulli" if bernoulli == True else "Gaussian"
        self.T = T

        self.inst = f"You are in a room with {len(arms)} buttons labeled {labels}.\n" \
        f"Each button is associated with a {self.bernoulli_or_gauss} distribution with a fixed but unknown mean; the means for the buttons could be different.\n" \
        "For each button, when you press it, you will get a reward that is sampled from the button's associated distribution.\n" \
        f"You have {T} time steps and, on each time step, you can choose any button and receive the reward.\n"

        if self.balance_EE:
            self.inst = self.inst + f"Your goal is to balance the explore-exploit trade-off and maximize the total reward over the {T} time steps.\n"
        else:
            self.inst = self.inst + f"Your goal is to maximize the total reward over the {T} time steps.\n"

        if self.end_with_goal == False:
            self.inst += "At each time step, I will show you your past choices and rewards. Then you must make\n" \
            f"the next choice, which must be exactly one of {self.labels}. You must\n" \
            "provide your final answer immediately within the tags <Answer>COLOR</Answer>\n" \
            f"where COLOR is one of {self.labels} and with no text explanation.\n"

        # all this becomes the system message
        self.messages = [{"role": "system", "content": self.inst}]

    # append a choice and reward to history
    def update(self, chosen_arm, a, reward):
        self.step += 1
        self.history += f"{chosen_arm} button, reward {reward}\n"

    def make_decision(self):

        # can't call this function before setup()
        assert self.messages is not None

        if self.step == 0:
            # first time step (no history)
            prompt = "So far you have played 0 times.\n\n"
        else:
            # subsequent time steps (w/ outcome history)
            if self.provide_context_on_round:
                prompt = f"So far you have played {self.step} out of your {self.T} time steps with the following choices and rewards:\n\n" + self.history + "\n\n"
            else:
                prompt = f"So far you have played {self.step} times with the following choices and rewards:\n\n" + self.history + "\n\n"

        prompt += "Which button will you choose next?\n"

        if self.end_with_goal == False:
            prompt += "Remember, YOU MUST provide your final"
        else:
            prompt += "You must provide your"

        prompt += " answer within the tags <Answer>COLOR</Answer>, " \
        f"where COLOR is one of {self.labels}."

        if self.short_answers:
            prompt += " Do not give an explanation."

        self.messages.append({"role": "user", "content": prompt})

        # get model's response and extract chosen arm

        # make a regex to extract the answer
        pattern = r"<Answer>(.*?)</Answer>"

        try:
            response = self.model(messages=self.messages, temperature=self.temperature)
            matches = re.findall(pattern, response)
            # take the first match as answer
            chosen_arm = matches[0].strip().lower()
        except IndexError:
#            print(f"An exception occured. Exiting early\n{traceback.format_exc()}")
            chosen_arm = "Not present"

        # if we ought to save this response, save the prompt and response in a file
        if self.save_every_nth_response != 0 and (self.step % self.save_every_nth_response) == 0:
            with open(f"{self.name}-step-{self.step}", 'w') as f:
                f.write(json.dumps(self.messages[0]))
                f.write(json.dumps(self.messages[1]))
                f.write(response)

        del self.messages[1]

        return (chosen_arm, response)


    def get_params(self):

        params = dict()

        params['type'] = "LLM"
        params['name'] = self.name
        params['model'] = self.model.model
        params['balance_EE'] = self.balance_EE
        params['short_answers'] = self.short_answers
        params['temperature'] = self.temperature
        params['end_with_goal'] = self.end_with_goal
        params['provide_context_on_round'] = self.provide_context_on_round

        return params
