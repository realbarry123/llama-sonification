import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelWrapper():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        self.chunk_size = 4
        self.next_seq_idx = 0
        self.next_state_idx = 0
        self.context = None
        self.hidden_states = None
    

    def seed(self, text):
        self.context = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        self.generate(self.chunk_size)


    def generate(self, tokens):
        #if not self.context:
        #    raise ValueError("must provide seed before generation")

        with torch.no_grad():
            output = self.model.generate(
                **self.context,
                max_new_tokens=tokens,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=False
            )
        
        self.context["input_ids"] = output.sequences
        self.context["attention_mask"] = torch.ones_like(output.sequences)
        self.hidden_states = output.hidden_states
    

    def next(self):
        if self.next_state_idx == self.chunk_size:
            self.generate(self.chunk_size)
            self.next_state_idx = 0
        
        if self.next_seq_idx-1 >= 0:
            next_token = self.context["input_ids"][0][self.next_seq_idx-1]
            next_token = self.tokenizer.decode(next_token)
        else:
            next_token = " "

        # Slice from the hidden states (rather than indexing)
        next_state = self.hidden_states[self.next_state_idx : self.next_state_idx+1]

        self.next_seq_idx += 1
        self.next_state_idx += 1
        return next_token, self.format_hidden_states(next_state)


    def get_last_tokens(self):

        if self.output == None:
            raise ValueError("output does not exist yet")
        
        generated_tokens = self.tokenizer.convert_ids_to_tokens(
            self.context[-1], 
            skip_special_tokens=True
        )
        return generated_tokens
    
    @staticmethod
    def format_hidden_states(hidden_states):
    
        steps = []

        for step in hidden_states:
            layers = torch.stack(
                [layer.squeeze(0)[-1] for layer in step]
            )
            steps.append(layers)

        return torch.stack(steps).to("cpu") # (time, layers, hidden)
    

if __name__ == "__main__":
    model = ModelWrapper()
    model.seed("I am")
    for i in range(10):
        token, state = model.next()
        print(token)
        print(state.shape)
        time.sleep(1)
    # model.generate(5)
    # print(model.get_last_output())
    # print(model.get_last_hidden_state().shape)