import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelWrapper():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        self.chunk_size = 1
        self.context_limit = 8
        self.next_seq_idx = 0
        self.next_state_idx = 0
        self.context = None
        self.history = ""
        self.hidden_states = None
        self.filler_token_id = self.tokenizer.encode("…", add_special_tokens=False)[0]
        self.trim_length = 4
    

    def seed(self, text):
        self.context = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        self._generate(self.chunk_size)
    
    
    def _trim_context(self):
        self.history += self.tokenizer.batch_decode(self.context["input_ids"][:, :self.trim_length])[0]

        self.context["input_ids"] = self.context["input_ids"][:, self.trim_length:]

        self.next_seq_idx -= self.trim_length


    def _generate(self, tokens):

        with torch.no_grad():
            output = self.model.generate(
                **self.context,
                max_new_tokens=tokens,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=False,
                temperature=0.7,
                eos_token_id=self.filler_token_id
            )
        
        self.context["input_ids"] = output.sequences
        # print(len(self.context["input_ids"][0]))
        if len(self.context["input_ids"][0]) >= self.context_limit:
            self._trim_context()
        self.context["attention_mask"] = torch.ones_like(output.sequences)
        self.hidden_states = output.hidden_states
    

    def next(self):
        if self.context == None:
            raise RuntimeError("model context not initialized. Did you call `seed`?")
        
        if self.next_state_idx == self.chunk_size:
            self._generate(self.chunk_size)
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
    

    @staticmethod
    def format_hidden_states(hidden_states):
    
        steps = []

        for step in hidden_states:
            layers = torch.stack(
                [layer.squeeze(0)[-1] for layer in step]
            )
            steps.append(layers)

        return torch.stack(steps).to("cpu") # (time, layers, hidden)
    

    def write_context(self, file_path: str):
        text = self.tokenizer.batch_decode(self.context["input_ids"])[0]
        if text[:17] == "<|begin_of_text|>":
            text = "\n" + text
        with open(file_path, "a") as f:
            f.write(text)


    def write_history(self, file_path: str):
        text = self.history + self.tokenizer.batch_decode(self.context["input_ids"])[0]
        if text[:17] == "<|begin_of_text|>":
            text = "\n" + text
        with open(file_path, "a") as f:
            f.write(text)
    

if __name__ == "__main__":
    model = ModelWrapper()
    model.seed("I am")
    try:
        while True:
            token, state = model.next()
            print(token)
    except KeyboardInterrupt:
        model.write_history("context.txt")
    # model.generate(5)
    # print(model.get_last_output())
    # print(model.get_last_hidden_state().shape)