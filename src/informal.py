# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.inference import inference
from datasets import load_dataset
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

# %% Testing inference function
system_prompt = "Be concise and answer the question directly. Do not provide any additional information."
user_prompt = "What is 2+2?"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").input_ids

# %%
output = inference(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    seed=42,
)

# %%
epsilon = 1e-9
token_distribution = output.token_distribution
token_log_distribution = (output.token_distribution + epsilon).log()

token_entropy = -torch.sum(token_distribution * token_log_distribution, dim=-1)
sequence_entropy = torch.mean(token_entropy, dim=-1)

sequence_entropy
