# Informal notebook to test stuff

# %%
%reload_ext autoreload
%autoreload 2

import gc
from datasets.utils.py_utils import Literal
import torch.nn.functional as F
import torch
from src.engine.scenarios.gsm import GSM8K
from transformers import pipeline

eps = 1e-8

dataset = GSM8K()

sample_result = dataset.sample(format="cot")

print("Prompt:", sample_result["prompt"])
print('')
print("Reference:", sample_result["reference"])

pipe = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    tokenizer="microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=False,
    device_map="cpu",
)

tokenizer = pipe.tokenizer
model = pipe.model

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
).to("cpu")

prompt_length = inputs.input_ids.shape[1]

# %%
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1024,
        temperature=0.5,
        num_return_sequences=2,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
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
def hidden_state_reshape(hidden_states: tuple[tuple[torch.Tensor]]) -> torch.Tensor:
    """
    Reshape the hidden states from the model output to a tensor of shape:
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


# Token-pooling function
def pool_uq_tokens(
    metric: torch.Tensor, 
    last_layer_distribution: torch.Tensor,
    sequences: torch.Tensor,
    pad_token_id: int,
    pooling_ratio: float = 1,
    weighting: None | Literal["entropy", "prob"] = None,
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

    print(output_mask)

    with torch.no_grad():
        # Normalize the divergence importance per token
        if weighting == "entropy":
            weights = -torch.sum(last_layer_distribution * torch.log(last_layer_distribution + 1e-8), dim=-1)

        elif weighting == "prob":
            max_probs, _ = torch.max(last_layer_distribution, dim=-1)
            weights = 1 - max_probs
        
        if weighting:
            metric = weights * metric # type: ignore

        # Pool the top largest divergences based on the ratio
        pool_amount = max(1, int(pooling_ratio * metric.shape[1]))
        top_k_values, top_k_ids = torch.topk(metric, pool_amount, dim=-1)

        if output_mask is not None:
            selected_mask_values = torch.gather(
                output_mask, 
                dim=-1, 
                index=top_k_ids
            )
            top_k_values = torch.where(selected_mask_values == 0, torch.nan, top_k_values)

        # Mean over remaining pool (ignoring padding/eos)
        result = torch.nanmean(top_k_values, dim=-1) 

    torch.cuda.empty_cache()
    gc.collect()

    return result

# %%
EarlyExitUQMetric = Literal[
    "state_mean_exit_layer",
    "softmax_mean_exit_layer",
]

def early_exit_uq(
    hidden_states, # [layer, batch_size, sequence_length, hidden_size] 
    lm_head,
    threshold: float,
    sequences: torch.Tensor,
    pad_token_id: int,
    metric_name: EarlyExitUQMetric,
    pooling_ratio=1,
    weighting: None | Literal["entropy", "prob"] = None,
):
    with torch.no_grad():
        last_layer_distribution = F.softmax(lm_head(hidden_states[-1]), dim=-1)

        layer_amount = hidden_states.shape[0]
        layer_uq_tensor = torch.zeros(
            hidden_states.shape[0], # layers 
            hidden_states.shape[1], # batch size
            hidden_states.shape[2], # sequence length
        ).to("cpu")

        for layer_idx in range(len(layer_amount) - 1):
            layer_states = hidden_states[layer_idx]
            next_layer_states = hidden_states[layer_idx + 1]

            if metric_name == "state_mean_exit_layer":
                difference = 1 - F.cosine_similarity(
                    layer_states,
                    next_layer_states,
                    dim=-1,
                    eps=1e-8
                )

            elif metric_name == "softmax_mean_exit_layer":
                layer_distribution = F.softmax(lm_head(layer_states), dim=-1)
                top_token_prob, _ = torch.max(layer_distribution, dim=-1) # [batch_size, sequence_length]

                next_layer_distribution = F.softmax(lm_head(next_layer_states), dim=-1)
                next_layer_top_token_prob, _ = torch.max(next_layer_distribution, dim=-1) # [batch_size, sequence_length]

                difference = torch.abs(top_token_prob - next_layer_top_token_prob)

            # [layers, batch_size, sequence_length]
            layer_uq_tensor[layer_idx] = difference < threshold

        token_uq = torch.argmax(layer_uq_tensor, dim=0)

        result = pool_uq_tokens(
            token_uq, # type: ignore
            last_layer_distribution,
            sequences,
            pad_token_id,
            pooling_ratio,
            weighting
        )

    return result

