# %%
# Environment smoke test

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# %%
device="mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# %%
# Load a small model (GPT-2
tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
model=GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Tokenizing a random input
text = "Once upon a time there was a bunny"
inputs = tokenizer(text, return_tensors="pt").to(device) # "pt" is. pytorch tesor

# performing inference
with torch.no_grad():
    outputs=model(**inputs, output_hidden_states=True) # the flag returns the outputs of each layer of the transformer

# Diagnostics
# Check we got activations
print(f"Number of layers: {len(outputs.hidden_states)}")
print(f"Hidden state shape: {outputs.hidden_states[-1].shape}")
print("Smoke test passed!")