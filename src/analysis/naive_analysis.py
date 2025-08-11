# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import jsonlines
import torch

# %% Analysis of GSM8K results
benchmark_src_path = "src/data/benchmark_results/"

with jsonlines.open(
    f"{benchmark_src_path}/gsm8k/samples_gsm8k_2025-08-11T10-35-08.100989.jsonl",
    mode="r",
) as reader:
    gsm8k_results = [item for item in reader]
    gsm8k_results = list(
        filter(
            lambda x: x["filter"] == "flexible-extract",
            gsm8k_results,
        )
    )


# %% Load tensors
def load_tensor(unique_id: str):
    tensor_path = f"{tensor_src_path}/{unique_id}.pt"
    return torch.load(tensor_path)


tensor_src_path = "src/data/tensor_states/"
unique_id = gsm8k_results[0]["prompt_hash"]

tensor_data = load_tensor(unique_id)
sample_data = gsm8k_results[0]
attentions = tensor_data["attentions"]
hidden_states = tensor_data["hidden_states"]

# %% Compute benchmark score
benchmark_score = sum(
    [int(result["exact_match"]) for result in gsm8k_results]
) / len(gsm8k_results)

print(benchmark_score)


# %% Compute output token distribution, sequence probabilities and attention matrix
attentions.shape
hidden_states.shape

# %%
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# %%
tokenized_response = tokenizer(sample_data["resps"][0][0], return_tensors="pt")
tokenized_response.input_ids.shape


# %%
@dataclass
class InternalState:
    # [batch_size, output_sequence_length, embedding_size]
    hidden_states: torch.Tensor

    # [batch_size, num_heads, output_sequence_length, full_sequence_length]
    attentions: torch.Tensor

    # [batch_size, output_sequence_length, top_k]
    token_distribution: torch.Tensor

    # [batch_size, output_sequence_length]
    token_distribution_ids: torch.Tensor

    # [batch_size, output_sequence_length]
    sequence_probabilities: torch.Tensor


def process_result(
    model_name: str,
    sample_data,
    hidden_states: torch.Tensor,
    attentions: torch.Tensor,
) -> InternalState:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get the probability distribution for each token in the sequence
    lm_head = model.lm_head
    hidden_states = hidden_states.to(lm_head.weight.dtype)

    logits = lm_head(hidden_states)
    token_distribution = torch.softmax(logits, dim=-1)

    # Get the output sequence probabilities
    generated_text = sample_data["resps"][0][0]
    tokenized_response = tokenizer(generated_text, return_tensors="pt")
    response_token_ids = tokenized_response.input_ids.squeeze(0)

    output_sequence_length = response_token_ids.shape[0]
    batch_size, sequence_length, _ = token_distribution.shape
    actual_length = min(output_sequence_length, sequence_length)

    sequence_probabilities = torch.zeros(batch_size, actual_length)
    for i in range(actual_length):
        if i < len(response_token_ids):
            token_id = response_token_ids[i]
            pos_in_full_seq = sequence_length - actual_length + i
            sequence_probabilities[0, i] = token_distribution[
                0, pos_in_full_seq, token_id
            ]

    # Prune attentions, hidden states and token distributions
    pruned_attentions = attentions[:, :, -actual_length:, :]
    truncated_hidden_states = hidden_states[:, -actual_length:, :]
    truncated_token_distribution = token_distribution[:, -actual_length:, :]


# %%
model_name = "gpt2"
internal_state = process_result(
    model_name,
    sample_data,
    hidden_states=hidden_states,
    attentions=attentions,
)

# %%
internal_state.sequence_probabilities
