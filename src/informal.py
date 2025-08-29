# Informal notebook to test stuff

# %%
# %reload_ext autoreload
# %autoreload 2
#
from datasets.utils.py_utils import Literal
import torch.nn.functional as F
import torch
from src.engine.scenarios.gsm import GSM8K
from transformers import pipeline

#%%
eps = 1e-8

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
def kl_divergence(probs_p: torch.Tensor, probs_q: torch.Tensor) -> torch.Tensor:
    """
    Input: probs_p [batch_size, sequence_length, vocab_size], probs_q idem
    Output: [batch_size, sequence_length]
    """

    probs_p = probs_p.clamp(min=eps)
    probs_q = probs_q.clamp(min=eps)
    kl = (probs_p * (probs_p.log() - probs_q.log())).sum(dim=-1)

    return kl

def mean_kl_divergence(
    hidden_states, 
    lm_head, 
    quarters_from_end=1,
    pooling_ratio=1,
    normalization: None | Literal["entropy", "max_prob"] = None,
):
    # (1/L) * Sum_i KL(p_i || p_L) (and then weighted average over pool ratio tokens)

    last_layer_distribution = F.softmax(lm_head(hidden_states[-1]))
    layer_amount = len(last_layer_distribution)

    kl_divergence_sum = torch.zeros(
        last_layer_distribution.shape[0], 
        last_layer_distribution.shape[1]
    )

    layer_from = int(layer_amount * (1 - quarters_from_end / 4))
    for layer_states in range(layer_from, layer_amount - 1):
        layer_distribution = F.softmax(lm_head(layer_states))
        layer_to_last_kl_div = kl_divergence(
            layer_distribution, 
            last_layer_distribution
        )

        kl_divergence_sum += layer_to_last_kl_div / (
            layer_amount * quarters_from_end / 4
        )

    # Normalize the divergence importance per token
    if normalization == "entropy":
        weights = -torch.sum(last_layer_distribution * torch.log(last_layer_distribution + 1e-8), dim=-1)

    elif normalization == "max_prob":
        max_probs, _ = torch.max(last_layer_distribution, dim=-1)
        weights = 1 - max_probs
    
    if normalization:
        kl_divergence_sum = weights * kl_divergence_sum # type: ignore

    # Pool the top largest divergences based on the ratio
    pool_amount = max(1, int(pooling_ratio * kl_divergence_sum.shape[1]))
    top_k_values, _ = torch.topk(kl_divergence_sum, pool_amount, dim=-1)

    result = torch.mean(top_k_values, dim=-1)

    return result 

def var_kl_divergence(hidden_states, lm_head, quarters_from_end=1):
    # Var(KL(p_1 || p_L), ..., KL(p_L-1 || p_L))
    pass

def var_shannon(hidden_states, lm_head, quarters_from_end=1):
    # Var(H(p_1), ..., H(p_L))
    pass

