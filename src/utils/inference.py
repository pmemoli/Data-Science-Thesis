from src.utils.types import InferenceOutput
import torch.nn.functional as F
import torch


def inference(
    model,
    tokenizer,
    messages: list[list[dict]],
    device: str = "auto",
    seed: int = 42,
    top_k: int = 30,
    on_hidden_states: bool = False,
) -> InferenceOutput:

    torch.manual_seed(seed)
    model.to(device)

    # Input
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(
        device
    )

    # Output generation
    extra_kwargs = {}
    if on_hidden_states:
        extra_kwargs["output_hidden_states"] = True

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
        **extra_kwargs,
    )

    # Generated sequence text and ids
    sequences = outputs.sequences
    prompt_length = inputs.shape[1]

    generated_ids = sequences[:, prompt_length:]
    generated_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )

    # Token probabilities for the generated sequence
    logits = torch.stack(outputs.scores, dim=1)
    probabilities = F.softmax(logits, dim=-1)

    token_probabilities = torch.gather(
        probabilities,
        -1,
        generated_ids.unsqueeze(-1),
    ).squeeze(-1)

    # Token distribution (for all layers if specified)
    if on_hidden_states:
        layers_idx = range(len(outputs.hidden_states[0]))  # type: ignore

        generatation_probabilities = []
        generation_ids = []
        for step, layer_outputs in enumerate(outputs.hidden_states):
            step_probabilities = []
            step_ids = []

            for layer in layers_idx:  # type: ignore
                if step != 0:
                    layer_hidden = layer_outputs[layer].squeeze(1)
                else:
                    layer_hidden = layer_outputs[layer][:, -1, :]

                logits = model.lm_head(layer_hidden)
                probabilities = F.softmax(logits, dim=-1)

                top_k_probabilities, top_k_ids = torch.topk(
                    probabilities, top_k, dim=-1
                )

                step_probabilities.append(top_k_probabilities)
                step_ids.append(top_k_ids)

            step_tensor = torch.stack(step_probabilities, dim=1)

            generatation_probabilities.append(step_tensor)
            generation_ids.append(torch.stack(step_ids, dim=1))

        # [batch_size, layers, sequence_length, top_k]
        token_distribution = torch.stack(generatation_probabilities, dim=2)
        token_distribution_ids = torch.stack(generation_ids, dim=2)

    else:
        # [batch_size, sequence_length, top_k]
        top_k_probabilities, top_k_ids = torch.topk(
            probabilities, top_k, dim=-1
        )

        # [batch_size, 1, sequence_length, top_k]
        top_k_probabilities = top_k_probabilities.unsqueeze(1)

        token_distribution = top_k_probabilities
        token_distribution_ids = top_k_ids.unsqueeze(1)

    # Generated sequence lengths
    eos_id = tokenizer.eos_token_id
    generated_tokens = torch.where(generated_ids != eos_id, 1, 0)
    sequence_length = generated_tokens.sum(dim=-1)

    return InferenceOutput(
        generated_ids=generated_ids,
        sequence_length=sequence_length,
        generated_text=generated_text,
        token_probabilities=token_probabilities,
        token_distribution=token_distribution,
        token_distribution_ids=token_distribution_ids,
    )
