# Informal notebook to test stuff

# %%
%reload_ext autoreload
%autoreload 2

import torch.nn.functional as F
import torch
from src.engine.scenarios.gsm import GSM8K
from transformers import pipeline

#%%
dataset = GSM8K()

#%%
sample_result = dataset.sample(format="cot")

print("Prompt:", sample_result["prompt"])
print('')
print("Reference:", sample_result["reference"])

#%%
pipe = pipeline(
    "text-generation",
    model="microsoft/Phi-3.5-mini-instruct",
    tokenizer="microsoft/Phi-3.5-mini-instruct",
    trust_remote_code=False,
    device_map="cuda:0",
)

#%%
tokenizer = pipe.tokenizer
model = pipe.model

#%% apply chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": sample_result["prompt"]},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Adds proper assistant prompt
    return_tensors="pt",
    return_dict=True,  # Returns dict with input_ids AND attention_mask
).to("cuda:0")

prompt_length = inputs.input_ids.shape[1]

# %%
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1024,
        temperature=0.5,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        output_hidden_states=True,
    )

# %%
sequences = outputs.sequences[:, prompt_length:]
for i, sequence in enumerate(sequences):
    generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
    print(f"Generated {i+1}:\n", generated_text)
    print('---')

# %%
def early_stop(hidden_states):
    pass

