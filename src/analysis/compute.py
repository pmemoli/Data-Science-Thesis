# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.metrics.token_uq.logits_uq import logits_uq
from src.metrics.sequence_ensemble import influence
import torch.nn.functional as F
import torch
import json
import hashlib


def hash_result(question, response):
    hasher = hashlib.sha256()
    hasher.update(question.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(response.encode("utf-8"))
    return hasher.hexdigest()


# %% Load data
tensor_data_filename = "src/data/runs/validation/gsm8k_microsoft_Phi-3.5-mini-instruct_20250916-210355.pt"
tensor_data = torch.load(
    tensor_data_filename, map_location=torch.device("cpu"), mmap=True
)

evaluation_data_filename = "src/data/runs/validation/evaluations_gsm8k_microsoft_Phi-3.5-mini-instruct_20250916-210355.json"
evaluation_data = json.load(
    open(evaluation_data_filename, "r", encoding="utf-8")
)

# %%
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

# %% Store shannon logit data
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
    is_correct = evaluation_data[hash]

    print(f"Processing item {len(data) + 1}/{len(tensor_data)}...")

    # Compute shannon entropy
    shannon_entropy = logits_uq(
        hidden_states=hidden_states,
        lm_head=model.lm_head,
        sequences=sequences,
        metric_name="logits_shannon_entropy",
    )  # [layer, batch_size, sequence_length]

    # Compute influence
    attention_influence_projection = influence(
        hidden_states=hidden_states,
        attentions=attentions,
        attention_outputs=attention_outputs,
        difference="projection",
    )  # [batch_size, sequence_length, sequence_length]

    attention_influence_angle = influence(
        hidden_states=hidden_states,
        attentions=attentions,
        attention_outputs=attention_outputs,
        difference="angle",
    )  # [batch_size, sequence_length, sequence_length]

    attention_influence_norm = influence(
        hidden_states=hidden_states,
        attentions=attentions,
        attention_outputs=attention_outputs,
        difference="norm",
    )  # [batch_size, sequence_length, sequence_length]

    data.append(
        {
            "prompt": prompt,
            "prompt_length": prompt_length,
            "generation": generation,
            "shannon_entropy": shannon_entropy,
            "angle_attention_influence": attention_influence_angle,
            "norm_attention_influence": attention_influence_norm,
            "projection_attention_influence": attention_influence_projection,
            "sequences": sequences,
            "is_correct": is_correct,
        }
    )
