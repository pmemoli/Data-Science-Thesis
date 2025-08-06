# Informal notebook to test stuff

# %%
%reload_ext autoreload
%autoreload 2

import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.metrics.entropy import shannon_entropy, predictive_entropy
from datasets import load_dataset
from src.utils.inference import inference
import torch

# %%
math_ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
gsm8k_ds = load_dataset("openai/gsm8k", "main", split="train")

# %%
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# %%
messages = [
    [
        {
            "role": "system",
            "content": "You are a helpful assistant who answers concisely",
        },
        {
            "role": "user",
            "content": "What is the integral of log(x) dx?",
        },
    ],
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt", padding=True).input_ids

#%%
outputs = model.generate(
    inputs,
    max_new_tokens=1000,
    do_sample=True,
    temperature=1.0,
    top_k=30,
    top_p=0.98,
    return_dict_in_generate=True,
    output_scores=True,
    pad_token_id=tokenizer.eos_token_id,
    output_attentions=True,
)

#%%
# [tokens, layers, batch_size, heads, gen_length, seq_length]
attentions = outputs.attentions 

# [batch_size, heads, input_seq_length, input_seq_length]
prefill_attention = outputs.attentions[0][-1]  

# list of tensors, each tensor is of shape [batch_size, heads, seq_length]
decode_attentions = [step_att[-1].squeeze(dim=2) for step_att in outputs.attentions[1:]]

token_entropy_list = [] # list of seq_len tensors [batch_size] 
for token_attention in decode_attentions:
    token_attention_log = -token_attention.log()
    token_head_entropy = (token_attention_log * token_attention).mean(dim=2)  
    token_mean_entropy = token_head_entropy.mean(dim=1)

    token_entropy_list.append(token_mean_entropy)

token_entropy = torch.stack(token_entropy_list, dim=1)  

attention_entropy = torch.mean(token_entropy, dim=1)  

#%%
generated_text = tokenizer.batch_decode(
    outputs.sequences, skip_special_tokens=True
)
print(generated_text)
