# Informal notebook to test stuff

# %%
import torch.nn.functional as F

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.metrics.entropy import shannon_entropy
from datasets import load_dataset
from src.utils.inference import inference

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
            "content": "Solve the following math problem: What is 2 + 2?",
        },
    ],
    [
        {
            "role": "system",
            "content": "You are a helpful assistant who answers concisely",
        },
        {
            "role": "user",
            "content": "Solve the following math problem: What is 2 + 5?",
        },
    ],
]

# %%
output = inference(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
)

# %%
output.token_distribution.size()


# %%
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt", padding=True).input_ids

# %% Output generation
top_k = 50
outputs = model.generate(
    inputs,
    max_new_tokens=1000,
    do_sample=True,
    temperature=1.0,
    top_k=top_k,
    top_p=0.98,
    return_dict_in_generate=True,
    output_scores=True,
    pad_token_id=tokenizer.eos_token_id,
    output_hidden_states=True,
)

# %%
sequences = outputs.sequences
prompt_length = inputs.shape[1]

generated_ids = sequences[:, prompt_length:]
generated_text = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True
)

# %%
# ouputs.hidden_states[step][layer]
# I want probabilities: [batch_size, layer, sequence_length, vocab_size]
generatation_probabilities = []
for step, layer_outputs in enumerate(outputs.hidden_states):
    step_probabilities = []

    for layer in range(len(outputs.hidden_states[step])):
        if step != 0:
            layer_hidden = layer_outputs[layer].squeeze(1)
        else:
            layer_hidden = layer_outputs[layer][:, -1, :]

        logits = model.lm_head(layer_hidden)
        probabilities = F.softmax(logits, dim=-1)  # [batch_size, vocab_size]

        step_probabilities.append(probabilities)

    # [batch_size, layers, vocab_size]
    step_tensor = torch.stack(step_probabilities, dim=1)
    generatation_probabilities.append(step_tensor)

# %%
# [batch_size, layers, sequence_length, vocab_size]
distribution_tensor = torch.stack(generatation_probabilities, dim=2)
distribution_tensor.size()  # [batch_size, layers, sequence_length, vocab_size]

# [batch_size, layer, sequence_length, vocab_size]
# distribution_tensor = torch.cat(generatation_probabilities, dim=2)

# layer_hidden = outputs.hidden_states[21][25]
# logits = model.lm_head(layer_hidden)
# probabilities = F.softmax(logits, dim=-1)
# probabilities.size() # [batch_size, 1, vocab_size]

# %%
# Generated sequence text and ids
sequences = outputs.sequences
prompt_length = inputs.shape[1]

generated_ids = sequences[:, prompt_length:]
generated_text = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True
)

# Token distribution
scores = outputs.scores
logits = torch.stack(scores, dim=1)
probabilities = F.softmax(logits, dim=-1)

# Selected token probabilities
indices = generated_ids.unsqueeze(-1)
token_probabilities = torch.gather(probabilities, 2, indices).squeeze(-1)

# Generated sequence lengths
eos_id = tokenizer.eos_token_id
generated_tokens = torch.where(generated_ids != eos_id, 1, 0)
sequence_length = generated_tokens.sum(dim=-1)
