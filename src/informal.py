# Informal notebook to test stuff

# %%
%reload_ext autoreload
%autoreload 2

import gc
from datasets.utils.py_utils import Literal
import torch.nn.functional as F
import torch
from src.engine.scenarios.gsm import GSM8K
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

eps = 1e-8

dataset = GSM8K()

sample_result = dataset.sample(format="cot")

print("Prompt:", sample_result["prompt"])
print('')
print("Reference:", sample_result["reference"])

# %%
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    attn_implementation="eager",
    trust_remote_code=False,
    device_map="cuda:0",
)
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=False,
)

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
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_attentions=True,
    )

# %%
sequences = outputs.sequences[:, prompt_length:]
for i, sequence in enumerate(sequences):
    generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
    print(f"Generated {i+1}:\n", generated_text)

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

# %%
def hidden_states_reshape(hidden_states: tuple[tuple[torch.Tensor]]) -> torch.Tensor:
    """
    Reshape the hidden states from the model output 
    [num_steps, num_layers, batch_size, 1 (except for encoding), hidden_size]

    to a tensor of shape:
    [num_layers, batch_size, generated_length, hidden_size]
    """
    num_steps = len(hidden_states)

    step_tensors = []

    for step_idx in range(0, num_steps):
        step_hidden_states = hidden_states[step_idx]
        step_hidden_states = torch.stack(step_hidden_states, dim=0) 

        if step_idx != 0:
            step_tensors.append(step_hidden_states)
        else:
            step_tensors.append(step_hidden_states[:, :, -1, :].unsqueeze(2))

    return torch.cat(step_tensors, dim=2) # type: ignore

# %%
def attentions_reshape(
    attentions: tuple[tuple[torch.Tensor]], 
    prompt_length: int
) -> torch.Tensor:
    """
    Reshape the attentions from the model output:
    [num_steps, num_layers] (tuple tuple)
    [batch_size, num_heads, 1 (except for encoding), seq_len_so_far]

    to a tensor of shape:
    [num_layers, batch_size, num_heads, generated_length, total_length]
    """

    num_steps = len(attentions)
    max_seq_len = attentions[-1][0].shape[-1]

    step_tensors = []

    # We ignore the encoding step (step_idx=0)
    for step_idx in range(1, num_steps):
        # [num_layers, batch_size, num_heads, 1, seq_len_so_far]
        step_attentions = attentions[step_idx]
        step_attentions = torch.stack(step_attentions, dim=0)

        # We want to pad to the right to have the same seq_len_so_far
        seq_len_so_far = step_attentions.shape[-1]
        pad_size = max_seq_len - seq_len_so_far
        step_attentions = F.pad(step_attentions, (0, pad_size), value=0.0)

        # Prune the prompt tokens
        step_tensors.append(step_attentions)

    return torch.cat(step_tensors, dim=3) # type: ignore

attentions = attentions_reshape(outputs.attentions, prompt_length)
attentions.shape

#%%
LogitsUQMetric = Literal[
    "logits_shannon_entropy",
    "logits_predictive_entropy",
    "logits_negative_log_likelihood"
]

def logits_uq(
    hidden_states, # [layer, batch_size, sequence_length, hidden_size] 
    lm_head, 
    sequences: torch.Tensor,
    metric_name: LogitsUQMetric,
):
    """
    Compute uncertainty metrics based on the last layer distribution.

    Input:
        hidden_states: tuple of tensors from the model, each of shape
                       [layer, batch_size, sequence_length, hidden_size]
        lm_head: the language model head to project hidden states to vocab size
        sequences: tensor of shape [batch_size, sequence_length] with token ids
        metric_name: one of "shannon_entropy", "predictive_entropy", "negative_log_likelihood"

    Output:
        token_uq: tensor of shape [batch_size, sequence_length] with uncertainty scores
    """

    with torch.no_grad():
        last_layer_distribution = F.softmax(lm_head(hidden_states[-1]), dim=-1)

        if metric_name == "logits_shannon_entropy":
            token_uq = -torch.sum(
                last_layer_distribution * torch.log(last_layer_distribution + eps), 
                dim=-1
            )

        elif metric_name == "logits_negative_log_likelihood":
            token_uq = -torch.log(torch.gather(
                last_layer_distribution, 
                dim=-1, 
                index=sequences.unsqueeze(-1)
            ).squeeze(-1) + eps)

        elif metric_name == "logits_predictive_entropy":
            selected_token_probs = torch.gather(
                last_layer_distribution, 
                dim=-1, 
                index=sequences.unsqueeze(-1)
            ).squeeze(-1)

            token_uq = -selected_token_probs * torch.log(selected_token_probs + eps)

    return token_uq

#%%
eps = 1e-8

# Token-pooling function
WeightMethod = Literal["entropy", "prob"]

def sequence_ensemble(
    metric: torch.Tensor,  # [batch_size, sequence_length]
    last_layer_distribution: torch.Tensor,
    sequences: torch.Tensor,
    pad_token_id: int,
    pooling_ratio: float = 1,
    weighting: None | WeightMethod = None,
):
    """
    Pool the uncertainty metric over the generated tokens based on different criterions.

    -metric: [batch_size, sequence_length]
    -last_layer_distribution: [batch_size, sequence_length, vocab_size]
    -output_mask: None or [batch_size, sequence_length] with 1 for valid tokens and 0 for padding/eos
    -pooling_ratio: float between 0 and 1 indicating the ratio of tokens to pool
    -weighting: None or "entropy" or "prob" to weight the metric by the token

    returns [batch_size]
    """

    output_mask = (sequences != pad_token_id).float() 

    with torch.no_grad():
        if weighting == "entropy":
            weights = -torch.sum(last_layer_distribution * torch.log(last_layer_distribution + 1e-8), dim=-1)

        elif weighting == "prob":
            max_probs, _ = torch.max(last_layer_distribution, dim=-1)
            weights = 1 - max_probs

        # And attention!

        pool_amount = max(1, int(pooling_ratio * metric.shape[1]))
        top_k_values, top_k_ids = torch.topk(metric, pool_amount, dim=-1)

        selected_mask_values = torch.gather(
            output_mask, 
            dim=-1, 
            index=top_k_ids
        )
        top_k_values = torch.where(selected_mask_values == 0, torch.nan, top_k_values)

        # Weighted average
        if weighting:
            selected_weights = torch.gather(weights, dim=-1, index=top_k_ids)
            selected_weights = torch.where(selected_mask_values == 0, torch.nan, selected_weights)
            
            result = torch.nansum(top_k_values * selected_weights, dim=-1) / torch.nansum(selected_weights, dim=-1)

        else:
            result = torch.nanmean(top_k_values, dim=-1)

    torch.cuda.empty_cache()
    gc.collect()

    return result

#%%
hidden_states = hidden_states_reshape(outputs.hidden_states)
token_logits_uq = logits_uq(
    hidden_states=hidden_states,
    lm_head=model.lm_head,
    sequences=sequences,
    metric_name="logits_shannon_entropy",
)

last_layer_distribution = F.softmax(model.lm_head(hidden_states[-1]), dim=-1)
pooled_uq = sequence_ensemble(
    metric=token_logits_uq,
    last_layer_distribution=last_layer_distribution,
    sequences=sequences,
    pad_token_id=tokenizer.pad_token_id,
    pooling_ratio=0.5,
    weighting=None,
)

pooled_uq
