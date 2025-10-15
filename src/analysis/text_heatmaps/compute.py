# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib.colors import Normalize, rgb2hex
from src.metrics.token_uq.logits_uq import logits_uq
from matplotlib import pyplot as plt
from typing import Literal
from matplotlib import cm
import string
import torch
import hashlib


def hash_result(question, response):
    hasher = hashlib.sha256()
    hasher.update(question.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(response.encode("utf-8"))
    return hasher.hexdigest()


# %% Load data
tensor_data_filename = "src/data/runs/gsm-exploration/gsm8k_microsoft_Phi-3.5-mini-instruct_20250916-210355.pt"
tensor_data = torch.load(
    tensor_data_filename, map_location=torch.device("cpu"), mmap=True
)

model_name = "microsoft/Phi-3.5-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=False,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=False,
)


# %% Rollout algorithms
eps = 1e-8


def attention_rollout(
    attentions: torch.Tensor,
    residual_stream_proportion: float = 0.5,
    attention_output_proportion: float = 0.5,
    receptive_field_norm: bool = False,
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

    dtype = attentions.dtype

    # Convert to specified dtype
    attentions = attentions.to(dtype=dtype)

    # Aggregate heads to [num_layers, batch_size, total_length, total_length]
    attentions = attentions.mean(dim=2)  # type: ignore

    # Weight by the amount of tokens that can attend to it
    if receptive_field_norm:
        n = attentions.size(-1)

        # Receptive field size for each column: [n, n-1, n-2, ..., 1]
        receptive_field = torch.arange(
            n, 0, -1, dtype=dtype, device=attentions.device
        )

        # Divide each column by its receptive field size
        attentions = attentions / receptive_field[None, None, None, :]

        # Renormalize rows
        attentions = attentions / attentions.sum(dim=-1, keepdim=True)

    # Normalize attention to take into account residual connections
    sequence_length = attentions.size(-1)
    identity = torch.eye(
        sequence_length, dtype=dtype, device=attentions.device
    )
    attentions = (
        attention_output_proportion * attentions
        + residual_stream_proportion * identity.unsqueeze(0).unsqueeze(0)
    )

    # Recursively multiply the weight matrices
    rollout = attentions[0, :, :, :]
    num_layers = attentions.size(0)
    for i in range(1, num_layers):
        rollout = torch.bmm(attentions[i, :, :, :], rollout)

    return rollout  # type: ignore


def influence(
    attentions: torch.Tensor,
    hidden_states: torch.Tensor,
    attention_outputs: torch.Tensor,
    difference: Literal["norm", "angle", "projection"] = "norm",
    receptive_field_norm: bool = False,
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

    dtype = attentions.dtype

    # Convert all inputs to specified dtype
    attentions = attentions.to(dtype=dtype)
    hidden_states = hidden_states.to(dtype=dtype)
    attention_outputs = attention_outputs.to(dtype=dtype)

    # Aggregate heads to [num_layers, batch_size, total_length, total_length]
    attentions = attentions.max(dim=2).values  # type: ignore

    # Weight by the amount of tokens that can attend to it
    if receptive_field_norm:
        n = attentions.size(-1)

        # Receptive field size for each column: [n, n-1, n-2, ..., 1]
        # receptive_field = torch.arange(
        #     n, 0, -1, dtype=dtype, device=attentions.device
        # )
        #
        # # Divide each column by its receptive field size
        # attentions = attentions / receptive_field[None, None, None, :]

        inverse_receptive_field = torch.arange(
            1, n + 1, dtype=dtype, device=attentions.device
        )
        # attentions = attentions * torch.sqrt(
        #     inverse_receptive_field[None, None, None, :] + eps
        # )
        attentions = attentions * inverse_receptive_field[None, None, None, :]

        # Renormalize rows
        attentions = attentions / attentions.sum(dim=-1, keepdim=True)

    # Normalize the attention matrix to take into account the residual connections
    num_layers = attentions.size(0)
    hidden_states = hidden_states[:-1]  # removes output hidden states

    if difference == "norm":
        # [num_layers, batch_size, total_length]
        hidden_state_norms = torch.linalg.norm(hidden_states, dim=-1)
        attention_output_norms = torch.linalg.norm(attention_outputs, dim=-1)

        # [num_layers, batch_size, total_length, total_length]
        hs_norm_matrix = torch.diag_embed(hidden_state_norms)
        ao_norm_matrix = torch.diag_embed(attention_output_norms)
        normalization_matrix = torch.diag_embed(
            1 / (attention_output_norms + hidden_state_norms + eps)
        )

        for layer_idx in range(num_layers):
            attentions[layer_idx] = torch.bmm(
                ao_norm_matrix[layer_idx], attentions[layer_idx]
            )
            attentions[layer_idx] = (
                hs_norm_matrix[layer_idx] + attentions[layer_idx]
            )
            attentions[layer_idx] = torch.bmm(
                normalization_matrix[layer_idx], attentions[layer_idx]
            )

    elif difference == "angle":
        sum = hidden_states + attention_outputs
        hs_angle = torch.cosine_similarity(hidden_states, sum, dim=-1)
        ao_angle = torch.cosine_similarity(attention_outputs, sum, dim=-1)

        hs_angle_matrix = torch.diag_embed(hs_angle)
        ao_angle_matrix = torch.diag_embed(ao_angle)
        normalization_matrix = torch.diag_embed(
            1 / (hs_angle + ao_angle + eps)
        )

        for layer_idx in range(num_layers):
            attentions[layer_idx] = torch.bmm(
                ao_angle_matrix[layer_idx], attentions[layer_idx]
            )
            attentions[layer_idx] = (
                hs_angle_matrix[layer_idx] + attentions[layer_idx]
            )
            attentions[layer_idx] = torch.bmm(
                normalization_matrix[layer_idx], attentions[layer_idx]
            )

    elif difference == "projection":

        def projection(a, b):
            return torch.sum(a * b, dim=-1) / (
                torch.linalg.norm(b, dim=-1) ** 2
            )

        sum = hidden_states + attention_outputs
        hs_proj = projection(hidden_states, sum)
        ao_proj = projection(attention_outputs, sum)

        hs_proj_matrix = torch.diag_embed(hs_proj)
        ao_proj_matrix = torch.diag_embed(ao_proj)
        normalization_matrix = torch.diag_embed(1 / (hs_proj + ao_proj + eps))

        for layer_idx in range(num_layers):
            attentions[layer_idx] = torch.bmm(
                ao_proj_matrix[layer_idx], attentions[layer_idx]
            )
            attentions[layer_idx] = (
                hs_proj_matrix[layer_idx] + attentions[layer_idx]
            )
            attentions[layer_idx] = torch.bmm(
                normalization_matrix[layer_idx], attentions[layer_idx]
            )

    # Influence algorithm
    influence = attentions[0, :, :, :]
    for i in range(1, num_layers):
        influence = torch.bmm(attentions[i, :, :, :], influence)

    return influence


# %%
def compute_macs(attentions, epsilon=0.87):
    """
    Compute Multi-Layer Attention Consistency Scores (MACS).

    Args:
        attentions: Shape [num_layers, batch, heads, seq_len, seq_len]
        input_length: Number of input tokens (N)
        epsilon: Floor vector coefficient (default: 0.05)

    Returns:
        z_scores: Token importance Z-scores, shape [batch, input_length]
        raw_consistency: Raw consistency scores, shape [batch, input_length]
    """
    if isinstance(attentions, list):
        attentions = torch.stack(attentions)

    num_layers = attentions.size(0)

    consistency = None
    for layer_idx in range(num_layers):
        # Max-pool across heads [batch, seq_len, seq_len]
        m_prime = attentions[layer_idx].max(dim=1).values

        # Apply floor vector
        m = (1 - epsilon) * m_prime + epsilon

        # Multiplicative aggregation
        consistency = m if consistency is None else consistency * m

    raw_consistency = consistency

    # Normalize to Z-scores
    mean = raw_consistency.mean(dim=-1, keepdim=True)
    std = raw_consistency.std(dim=-1, keepdim=True)
    z_scores = (raw_consistency - mean) / (std + 1e-8)
    z_scores = z_scores - z_scores.min(dim=-1, keepdim=True).values + 1e-8

    return z_scores, raw_consistency


# %%
def aggregate_attention_influence(influence: torch.Tensor) -> torch.Tensor:
    n = influence.size(-1)
    aggregated_influence = influence.sum(dim=0)
    divisors = torch.arange(
        n, 0, -1, dtype=influence.dtype, device=influence.device
    )
    aggregated_influence = aggregated_influence / divisors

    return aggregated_influence


def visualize_tuples(tuples):
    values = [v for _, v in tuples]

    norm = Normalize(vmin=min(values), vmax=max(values))
    cmap = cm.get_cmap("Blues")  # you can try "viridis", "coolwarm", etc.

    html_parts = []
    for s, v in tuples:
        color = rgb2hex(cmap(norm(v) * 0.9))
        html_parts.append(f'<span style="background-color:{color}">{s}</span>')

    html_str = "".join(html_parts)

    return html_str


def save_text_heatmap(
    filename: str,
    values: list[torch.Tensor],
    values_names: list[str],
    token_sequence: list[int],
):
    html_str = ""

    tokens = [
        tokenizer.convert_ids_to_tokens(token_id)
        for token_id in token_sequence
    ]
    tokens = tokens[0]

    special_tokens = {"<0x0A>", "<|end|>"}
    punctuation = set(string.punctuation) | {"，", "。", "、", "！", "？"}

    for i, influence in enumerate(values):
        influence_name = values_names[i]

        max_inf = max(influence).item()
        min_inf = min(influence).item()

        influence = torch.clamp(influence, 0, max=0.9 * max_inf)
        influence = (influence - min_inf) / (max_inf - min_inf + 1e-8)

        html_tokens = []
        for token, value in zip(tokens, influence):
            clean_token = token.replace("▁", " ")

            # Rule 1: Ignore special tokens (set to 0)
            if token in special_tokens or clean_token.strip() == "<0x0A>":
                value = torch.tensor(0)

            # Rule 2: If token is punctuation (single char in punctuation set), set to 0
            elif (
                clean_token.strip() in [".", ","] or clean_token in punctuation
            ):
                value = torch.tensor(0)

            html_tokens.append((clean_token, value.item()))

        html_str += f"<h2>{influence_name}</h2>"
        html_str += f"<div>{visualize_tuples(html_tokens)}</div>"

    with open(filename, "w") as f:
        f.write(html_str)


data = []
for item in tensor_data:
    prompt = item["prompt"]
    prompt_length = item["prompt_length"]
    generation = item["generation"]

    hidden_states = item["hidden_states"]
    attentions = item["attentions"]
    attention_outputs = item["attention_outputs"]
    sequences = item["sequences"]

    hash = hash_result(item["prompt"], item["generation"])

    if (
        hash
        != "16dab01c9bdba57f7166d1121849c7e5bff45a6d460ef3d605e66685a7b54ef2"
    ):
        continue

    heat_values = []
    values_names = ["shannon entropy", "influence", "rollout", "macs"]

    index = 10
    shannon_entropy = logits_uq(
        hidden_states=hidden_states,
        lm_head=model.lm_head,
        sequences=sequences,
        metric_name="logits_shannon_entropy",
    )[0]

    influence_values = influence(
        attentions,
        hidden_states,
        attention_outputs,
        difference="projection",
        receptive_field_norm=True,
    )[0]

    rollout_values = attention_rollout(
        attentions,
        attention_output_proportion=0.1,
        residual_stream_proportion=0.9,
        receptive_field_norm=True,
    )[0]

    zscores, consistency = compute_macs(attentions)
    macs = zscores[0].max(dim=0).values
    # macs = aggregate_attention_influence(zscores[0])
    macs = zscores[0, -1]

    heat_values.append(shannon_entropy[prompt_length:-index])
    heat_values.append(
        influence_values.max(dim=0).values[prompt_length:-index]
    )
    heat_values.append(rollout_values.max(dim=0).values[prompt_length:-index])
    heat_values.append(macs[prompt_length:-index])

    token_sequence = sequences[0, :-index].tolist()

    save_text_heatmap(
        f"src/analysis/text_heatmaps/heatmaps/{hash}_last.html",
        heat_values,
        values_names,
        sequences,
    )

    break
