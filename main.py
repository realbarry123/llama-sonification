import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonification import sonify, gainson
from data import normalize, pca_reduce
from scipy.io import wavfile
from plot import histogram
from timer import Timer

NEW_TOKENS = 8

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

tensor = torch.stack(steps) # (steps, layers, hidden)

time, layers, hidden = tensor.shape
audio_tensor = tensor.reshape(time * layers, hidden).float() # (time, voices)
#audio_tensor = pca_reduce(audio_tensor, q=128)
for n in range(9, 17):
    REPS = 8
    sum = 0
    for i in range(REPS):
        audio_tensor = torch.randn(audio_tensor.shape)
        timer = Timer()
        wav = sonify(audio_tensor[:(17*(n))], 0.12, do_interpolate=True, do_stereo=True, do_diff=False).numpy()
        sum += timer.get_time()
    print(f"{n}: {sum/REPS}")
#wav = gainson(audio_tensor, torch.arange(50, 2098), 0.12, do_stereo=True).numpy()

# wavfile.write("03-20.3_uniform-interpolate.wav", 44100, wav)
print("\a")