import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonification import sonify, normalize
from scipy.io import wavfile

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

inputs = tokenizer("riverrun, past Eve and Adams, from swerve of shore", return_tensors="pt").to(model.device)


with torch.no_grad(): 
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

hidden_states = outputs.hidden_states[-1].float().squeeze(0)
hidden_states = normalize(hidden_states, 50, 2050)

print(hidden_states.shape)

wavfile.write("02-23.0.wav", 44100, sonify(hidden_states[:, :], 0.2, do_stereo=True, do_diff=True))