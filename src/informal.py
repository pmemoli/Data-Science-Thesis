# Informal notebook to test stuff

# %%
from transformer_lens import HookedTransformer
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
print("")
print("Reference:", sample_result["reference"])

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
attention_outputs = {}


def save_attention_output(layer_idx):
    def hook(module, input, output):
        # Store attention output for this generation step
        if f"layer_{layer_idx}" not in attention_outputs:
            attention_outputs[f"layer_{layer_idx}"] = []
        attention_outputs[f"layer_{layer_idx}"].append(output[0].detach())

    return hook


for i, layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_hook(save_attention_output(i))

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
attention_outputs["layer_1"][4].shape


# %%
def hidden_states_reshape(
    hidden_states: tuple[tuple[torch.Tensor]],
) -> torch.Tensor:
    """
    Reshape the hidden states from the model output
    [num_steps][num_layers][batch_size, 1 (except for encoding), hidden_size]

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

    return torch.cat(step_tensors, dim=2)  # type: ignore


def attention_outputs_reshape(attention_outputs: dict) -> torch.Tensor:
    """
    Reshape the attention outputs from the hooks
    [layer_name][num_steps][batch_size, num_heads, 1 (except for encoding), hidden_size]

    To a tensor of shape
    [num_layers, batch_size, generated_length, hidden_size]
    """

    # With this, i can properly combine everything! Tomorrow i'm implementing this properly.

    num_steps = len(attention_outputs["layer_0"])

    step_tensors = []
    for step_idx in range(num_steps):
        step_hidden_states = []
        for layer_idx in range(len(attention_outputs.keys())):
            step_hidden_states.append(
                attention_outputs[f"layer_{layer_idx}"][step_idx]
            )

        step_hidden_states = torch.stack(step_hidden_states, dim=0)

        if step_idx != 0:
            step_tensors.append(step_hidden_states)
        else:
            step_tensors.append(step_hidden_states[:, :, -1, :].unsqueeze(2))

    return torch.cat(step_tensors, dim=2)  # type: ignore


# %%
hidden_states = hidden_states_reshape(outputs.hidden_states)

# %%
attention_outputs = attention_outputs_reshape(attention_outputs)

# GOD, solo queda ver como combinar. Estaria bueno plantear bien el algoritmo de attention rollow pesado.


# %%
def attentions_reshape(attentions: tuple[tuple[torch.Tensor]]) -> torch.Tensor:
    """
    Reshape the attentions from the model output:
    [num_steps, num_layers] (tuple tuple)
    [batch_size, num_heads, 1 (except for encoding), seq_len_so_far]

    to a tensor of shape:
    [num_layers, batch_size, num_heads, generated_length, total_length]
    """

    num_steps = len(attentions)
    max_seq_len = attentions[-1][0].shape[-1]

    # Encoding attentions
    encoding_attentions = attentions[0]
    encoding_attentions = torch.stack(encoding_attentions, dim=0)

    seq_len_so_far = encoding_attentions.shape[-1]
    pad_size = max_seq_len - prompt_length

    # [num_layers, batch_size, num_heads, prompt_length, total_length]
    encoding_attentions = F.pad(encoding_attentions, (0, pad_size), value=0.0)

    # Decoding attentions
    step_tensors = []
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

    decoding_attentions = torch.cat(step_tensors, dim=3)  # type: ignore

    full_attentions = torch.cat(
        [encoding_attentions, decoding_attentions], dim=3
    )

    return full_attentions  # type: ignore


def attention_rollout(
    attentions: torch.Tensor,
) -> torch.Tensor:
    """
    Perform attention rollout as described in the paper:
    https://aclanthology.org/2020.acl-main.385.pdf

    Input:
    attentions (torch.Tensor): Tensor of shape
    [num_layers, batch_size, num_heads, total_length, total_length]

    Output:
    torch.Tensor: Tensor of shape
    [batch_size, total_length, total_length]
    """

    # Aggregate heads to [num_layers, batch_size, total_length, total_length]
    attentions = attentions.mean(dim=2)  # type: ignore

    # Normalize by the receptive field size
    sequence_length = attentions.size(-1)
    receptive_field_sizes = (
        (
            torch.arange(sequence_length + 1, 1, step=-1)
            .float()
            .to(attentions.device)
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(attentions.device)

    attentions = attentions / (receptive_field_sizes + eps)
    attentions = attentions / (attentions.sum(dim=-1, keepdim=True) + eps)

    # Normalize attention to take into account residual connections
    identity = torch.eye(sequence_length).to(attentions.device)
    attentions = 0.5 * (attentions + identity.unsqueeze(0).unsqueeze(0))

    # Recursively multiply the weight matrices
    rollout = attentions[0, :, :, :]
    num_layers = attentions.size(0)
    for i in range(1, num_layers):
        rollout = torch.bmm(rollout, attentions[i, :, :, :])

    return rollout  # type: ignore


def influence(
    attentions: torch.Tensor,
) -> torch.Tensor:
    """
    Perform attention rollout as described in the paper:
    https://aclanthology.org/2020.acl-main.385.pdf

    Input:
    attentions (torch.Tensor): Tensor of shape
    [num_layers, batch_size, num_heads, total_length, total_length]

    Output:
    torch.Tensor: Tensor of shape
    [batch_size, total_length, total_length]
    """

    # Aggregate heads to [num_layers, batch_size, total_length, total_length]
    attentions = attentions.mean(dim=2)  # type: ignore

    # Normalize attention to take into account residual connections
    sequence_length = attentions.size(-1)
    identity = torch.eye(sequence_length).to(attentions.device)

    # In here, i want to weight the identity by the token embedding size in the corresponding laye, and the attentions by the VALUE norm.
    attentions = 0.1 * attentions + 0.9 * identity.unsqueeze(0).unsqueeze(0)

    # Recursively multiply the weight matrices
    rollout = attentions[0, :, :, :]
    num_layers = attentions.size(0)
    for i in range(1, num_layers):
        rollout = torch.bmm(rollout, attentions[i, :, :, :])

    return rollout  # type: ignore


# %%
attentions = attentions_reshape(outputs.attentions)
rollout = influence(attentions)

rollout[0][-2]

# # %%
# # 2 formas de agregar el rollout: mean y ult token
# WeightMethod = Literal[
#     "entropy", "prob", "attention_rollout_mean", "attention_rollout_last"
# ]
#
#
# def sequence_ensemble(
#     metric: torch.Tensor,  # [batch_size, sequence_length]
#     last_layer_distribution: torch.Tensor,
#     attentions: torch.Tensor,
#     sequences: torch.Tensor,
#     pad_token_id: int,
#     pooling_ratio: float = 1,
#     prompt_length: int = 0,
#     weighting: None | WeightMethod = None,
# ):
#     """
#     Pool the uncertainty metric over the generated tokens based on different criterions.
#
#     -metric: [batch_size, sequence_length]
#     -last_layer_distribution: [batch_size, sequence_length, vocab_size]
#     -output_mask: None or [batch_size, sequence_length] with 1 for valid tokens and 0 for padding/eos
#     -pooling_ratio: float between 0 and 1 indicating the ratio of tokens to pool
#     -weighting: None or "entropy" or "prob" to weight the metric by the token
#
#     returns [batch_size]
#     """
#
#     output_mask = (sequences != pad_token_id).float()
#
#     with torch.no_grad():
#         if not weighting:
#             pass
#
#         elif weighting == "entropy":
#             weights = -torch.sum(
#                 last_layer_distribution
#                 * torch.log(last_layer_distribution + 1e-8),
#                 dim=-1,
#             )
#
#         elif weighting == "prob":
#             max_probs, _ = torch.max(last_layer_distribution, dim=-1)
#             weights = 1 - max_probs
#
#         elif "attention_rollout" in weighting:
#             rollout = attention_rollout(attentions)
#             if "mean" in weighting:
#                 weights = torch.mean(rollout[:, :, prompt_length:], dim=-1)
#
#             elif "last" in weighting:
#                 weights = rollout[:, -1, prompt_length:]
#
#         pool_amount = max(1, int(pooling_ratio * metric.shape[1]))
#         top_k_values, top_k_ids = torch.topk(metric, pool_amount, dim=-1)
#
#         selected_mask_values = torch.gather(
#             output_mask, dim=-1, index=top_k_ids
#         )
#         top_k_values = torch.where(
#             selected_mask_values == 0, torch.nan, top_k_values
#         )
#
#         # Weighted average
#         if weighting:
#             selected_weights = torch.gather(weights, dim=-1, index=top_k_ids)
#             selected_weights = torch.where(
#                 selected_mask_values == 0, torch.nan, selected_weights
#             )
#
#             result = torch.nansum(
#                 top_k_values * selected_weights, dim=-1
#             ) / torch.nansum(selected_weights, dim=-1)
#
#         else:
#             result = torch.nanmean(top_k_values, dim=-1)
#
#     return result
#
#
# eps = 1e-8
#
# # Last layer distribution based metrics
# LogitsUQMetric = Literal[
#     "logits_shannon_entropy",
#     "logits_predictive_entropy",
#     "logits_negative_log_likelihood",
# ]
#
#
# def logits_uq(
#     hidden_states,  # [layer, batch_size, sequence_length, hidden_size]
#     lm_head,
#     sequences: torch.Tensor,
#     metric_name: LogitsUQMetric,
# ):
#     """
#     Compute uncertainty metrics based on the last layer distribution.
#
#     Input:
#         hidden_states: tuple of tensors from the model, each of shape
#                        [layer, batch_size, sequence_length, hidden_size]
#         lm_head: the language model head to project hidden states to vocab size
#         sequences: tensor of shape [batch_size, sequence_length] with token ids
#         metric_name: one of "shannon_entropy", "predictive_entropy", "negative_log_likelihood"
#
#     Output:
#         token_uq: tensor of shape [batch_size, sequence_length] with uncertainty scores
#     """
#
#     with torch.no_grad():
#         last_layer_distribution = F.softmax(lm_head(hidden_states[-1]), dim=-1)
#
#         if metric_name == "logits_shannon_entropy":
#             token_uq = -torch.sum(
#                 last_layer_distribution
#                 * torch.log(last_layer_distribution + eps),
#                 dim=-1,
#             )
#
#         elif metric_name == "logits_negative_log_likelihood":
#             token_uq = -torch.log(
#                 torch.gather(
#                     last_layer_distribution,
#                     dim=-1,
#                     index=sequences.unsqueeze(-1),
#                 ).squeeze(-1)
#                 + eps
#             )
#
#         elif metric_name == "logits_predictive_entropy":
#             selected_token_probs = torch.gather(
#                 last_layer_distribution, dim=-1, index=sequences.unsqueeze(-1)
#             ).squeeze(-1)
#
#             token_uq = -selected_token_probs * torch.log(
#                 selected_token_probs + eps
#             )
#
#     return token_uq
#
#
# # %%
# hidden_states = hidden_states_reshape(outputs.hidden_states)
#
# token_uq = logits_uq(
#     hidden_states=hidden_states,
#     lm_head=model.lm_head,
#     sequences=sequences[:, prompt_length:],
#     metric_name="logits_shannon_entropy",
# )
#
# last_layer_distribution = F.softmax(
#     model.lm_head(hidden_states[-1]), dim=-1
# ).to("cuda:0")
#
# pooled_uq = sequence_ensemble(
#     metric=token_uq,
#     last_layer_distribution=last_layer_distribution,
#     attentions=attentions_reshape(outputs.attentions),
#     sequences=sequences,
#     pad_token_id=tokenizer.pad_token_id,
#     weighting="attention_rollout_last",
#     prompt_length=prompt_length,
# )
#
# # Ok, tomorrow es ver los resultados con la nueva metrica...
# # Aunque 100% rollout tiene muchas limitaciones, va a ameritar ver de usar algo mas especifico para decoder only models

# %%
hidden_states = hidden_states_reshape(outputs.hidden_states)

# %%
(hidden_states[-1, 0, -19, :]).norm(dim=-1)
# hidden_states.shape
