import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonification import sonify, normalize
from scipy.io import wavfile

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

inputs = tokenizer("I am ", return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=8,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_scores=False
    )

# Generated token ids (prompt + new tokens)
generated_ids = output.sequences
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)

hidden_states = output.hidden_states # (step, layers, batch, seq, hidden)

steps = []

for step in hidden_states:

    layers = torch.stack(
        [layer.squeeze(0)[-1] for layer in step]
    )  # (layers, hidden)

    steps.append(layers)

tensor = torch.stack(steps) # (steps, layers, hidden)

time, layers, hidden = tensor.shape
audio_tensor = tensor.reshape(time * layers, hidden).float() # (time, voices)
audio_tensor = normalize(audio_tensor, 50, 2050)
wavfile.write("02-23.6_multstep-nodiff.wav", 44100, sonify(audio_tensor[:, :], 0.1, do_interpolate=True, do_stereo=True, do_diff=False))
