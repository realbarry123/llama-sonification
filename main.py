import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonifier import Sonifier
from scipy.io import wavfile

NEW_TOKENS = 3

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

inputs = tokenizer("I am", return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=NEW_TOKENS,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_scores=False
    )

# Generated token ids (prompt + new tokens)
generated_ids = output.sequences
#generated_text = tokenizer.convert_ids_to_tokens(generated_ids[0], skip_special_tokens=True)
#print(generated_text)

hidden_states = output.hidden_states 
# (step(tuple), layers(tuple), batch(tensor), seq(tensor), hidden(tensor))

# Copy/reshape select hidden states to steps
steps = []
for step in hidden_states:
    layers = torch.stack(
        [layer.squeeze(0)[-1] for layer in step]
    )  # (layers, hidden)

    steps.append(layers)

states = torch.stack(steps) # (time, layers, hidden)

sonify = Sonifier(states.shape)
wav = sonify(states).numpy()
wavfile.write("stereo.wav", 44100, wav)