import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonifier import Sonifier
from scipy.io import wavfile

NEW_TOKENS = 4

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
generated_text = tokenizer.convert_ids_to_tokens(generated_ids[0], skip_special_tokens=True)
print("\n".join(generated_text))

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


sonify = Sonifier(states.shape, note_length=2/17, fs=44100)
# sonify.config["freq_map"] = torch.arange(0, 2048)
sonify.config["sonification_type"] = "freq"
sonify.config["freq_lower"] = 0
wav = sonify(states).numpy()
wavfile.write("03-39.0.1_rand-init-phase.wav", 44100, wav)