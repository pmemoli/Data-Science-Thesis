import torch.nn.functional as F
import torch


def hidden_states_reshape(
    hidden_states: tuple[tuple[torch.Tensor]],
) -> torch.Tensor:
    """
    Reshape the hidden states from the model output
    [num_steps][num_layers][batch_size, 1 (except for encoding), hidden_size]

    to a tensor of shape:
    [num_layers, batch_size, total_length, hidden_size]
    """
    num_steps = len(hidden_states)

    step_tensors = []

    for step_idx in range(0, num_steps):
        step_hidden_states = hidden_states[step_idx]
        step_hidden_states = torch.stack(step_hidden_states, dim=0)

        step_tensors.append(step_hidden_states)

    return torch.cat(step_tensors, dim=2)  # type: ignore


def attention_outputs_reshape(attention_outputs: dict) -> torch.Tensor:
    """
    Reshape the attention outputs from the hooks
    [layer_name][num_steps][batch_size, num_heads, 1 (except for encoding), hidden_size]

    To a tensor of shape
    [num_layers, batch_size, total_length, hidden_size]
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

        step_tensors.append(step_hidden_states)

    return torch.cat(step_tensors, dim=2)  # type: ignore


def attentions_reshape(attentions: tuple[tuple[torch.Tensor]]) -> torch.Tensor:
    """
    Reshape the attentions from the model output:
    [num_steps, num_layers] (tuple tuple)
    [batch_size, num_heads, 1 (except for encoding), seq_len_so_far]

    to a tensor of shape:
    [num_layers, batch_size, num_heads, total_length, total_length]
    """

    num_steps = len(attentions)
    max_seq_len = attentions[-1][0].shape[-1]

    # Encoding attentions
    encoding_attentions = attentions[0]
    encoding_attentions = torch.stack(encoding_attentions, dim=0)

    seq_len_so_far = encoding_attentions.shape[-1]
    pad_size = max_seq_len - seq_len_so_far

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
