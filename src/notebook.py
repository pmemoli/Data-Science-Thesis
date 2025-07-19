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
            "content": "Solve the following math problem What is 2 + 2?",
        },
    ],
]

#%%
output = inference(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    device="cpu",
    on_hidden_states=True,
)

#%%
output.token_distribution.size()


# %%
shannon_entropy(
    token_distribution=output.token_distribution,
    sequence_length=output.sequence_length,
    layer=-1,
)

# %%
predictive_entropy(
    token_probabilities=output.token_probabilities,
    sequence_length=output.sequence_length,
)
