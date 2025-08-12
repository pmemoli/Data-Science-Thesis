# Informal notebook to test stuff

# %%
%reload_ext autoreload
%autoreload 2

import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# %%
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype="auto",
    attn_implementation="eager",
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
            "content": "What is 2+2?",
        },
    ],
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt", padding=True).input_ids

#%%
outputs = model.generate(
    inputs,
    max_new_tokens=1000,
    do_sample=True,
    temperature=1.0,
    top_k=30,
    top_p=0.98,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True,
    output_attentions=True,
    output_hidden_states=True,
    use_cache=True,
)

#%%
outputs.attentions

#%%
def _process_outputs(self, outputs, prompt_length, top_k=20):
    # Obtain the probability distribution of the generated tokens
    generated_ids = outputs.sequences[:, prompt_length:]

    logits = torch.stack(outputs.scores, dim=1)
    token_distribution = F.softmax(logits, dim=-1)

    topk_probs, topk_indices = torch.topk(
        token_distribution, k=top_k, dim=-1
    )

    # Obtain sequence probabilities
    sequence_probabilities = torch.gather(
        token_distribution,
        -1,
        generated_ids.unsqueeze(-1),
    ).squeeze(-1)

    # Obtain attention
    decode_attentions = [step_att[-1].squeeze(dim=2) for step_att in outputs.attentions[1:]]

    max_seq_length = max(att.size(-1) for att in decode_attentions)
    attention_tensor = torch.stack(
        [F.pad(att, (0, max_seq_length - att.size(-1))) for att in decode_attentions],
        dim=-2
    )

    # Obtain hidden states
    decode_hidden_states = [step_hidden[-1].squeeze(dim=1) for step_hidden in outputs.hidden_states[1:]]

    hidden_states_tensor = torch.stack(
        decode_hidden_states, dim=1
    )

    return {
        "hidden_states": hidden_states_tensor,
        "attentions": attention_tensor,
        "token_distribution": topk_probs,
        "token_distribution_ids": topk_indices,
        "sequence_probabilities": sequence_probabilities,
    }

#%%
result = _process_outputs(model, outputs, inputs.shape[1], top_k=50)

#%%
outputs.attentions

# %% Attention
decode_attentions = [step_att[-1].squeeze(dim=2) for step_att in outputs.attentions[1:]]

max_seq_length = max(att.size(-1) for att in decode_attentions)
attention_tensor = torch.stack(
    [F.pad(att, (0, max_seq_length - att.size(-1))) for att in decode_attentions],
    dim=-2
)

attention_tensor.shape

outputs.attentions[0][-1].shape

inputs.shape

#%%
generated_ids = outputs.sequences[:, inputs.shape[1]:]
generated_ids.shape

# decode
decoded_text = tokenizer.batch_decode(
    generated_ids
)

print(decoded_text)

#%%
generated_text = tokenizer.batch_decode(
    outputs.sequences, skip_special_tokens=True
)
print(generated_text)
