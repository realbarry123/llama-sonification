import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonification import sonify, gainson
from data import normalize, pca_reduce
from scipy.io import wavfile
from plot import histogram

NEW_TOKENS = 12

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
print(generated_text)

hidden_states = output.hidden_states 
# (step(tuple), layers(tuple), batch(tensor), seq(tensor), hidden(tensor))

# Copy/reshape select hidden states to steps
steps = []
for step in hidden_states:
    layers = torch.stack(
        [layer.squeeze(0)[-1] for layer in step]
    )  # (layers, hidden)

    steps.append(layers)

tensor = torch.stack(steps) # (steps, layers, hidden)

time, layers, hidden = tensor.shape
audio_tensor = tensor.reshape(time * layers, hidden).float() # (time, voices)
# audio_tensor = pca_reduce(audio_tensor, q=8)
audio_tensor = normalize(audio_tensor, 50, 1050)

wav = sonify(audio_tensor, 0.12, do_interpolate=False, do_stereo=True, do_diff=False).numpy()
wav = gainson(audio_tensor, torch.arange(100, 2148), 0.12, do_stereo=True).numpy()
wavfile.write("03-16.0_gainson.wav", 44100, wav)
print("\a")