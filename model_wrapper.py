import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelWrapper():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        self.chunk_size = 4
        self.next_chunk_idx = self.chunk_size
    

    def seed(self, text):
        self.context_str = text
        self.chunks_generated = 0


    def generate(self, tokens):
        if not self.context_str:
            raise ValueError("must provide seed before generation")

        context = self.tokenizer(self.context_str, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            self.output = self.model.generate(
                **context,
                max_new_tokens=tokens,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=False
            )
        
        self.context_str = self.tokenizer.decode(
            self.output.sequences[0], 
            skip_special_tokens=True
        )
        print(self.context_str)
        self.chunks_generated += 1
    
    def next(self):
        if self.next_chunk_idx == self.chunk_size: 
            self.generate(self.chunk_size)
            self.next_chunk_idx = 0
        token_idx = self.chunk_size * (self.chunks_generated - 1) + self.next_chunk_idx
        next_token = self.get_last_output()[token_idx]
        next_state = self.get_last_hidden_state()[token_idx]

        self.next_chunk_idx += 1
        return next_token, next_state

    def get_last_output(self):

        if self.output == None:
            raise ValueError("output does not exist yet")
        generated_ids = self.output.sequences
        generated_text = self.tokenizer.convert_ids_to_tokens(generated_ids[0], skip_special_tokens=True)
        return generated_text
    

    def get_last_hidden_state(self):
        
        if self.output == None:
            raise ValueError("output does not exist yet")
        
        hidden_states = self.output.hidden_states 

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