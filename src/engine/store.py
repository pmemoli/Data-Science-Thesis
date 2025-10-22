# Stores token-level activations and outputs during generation

from transformers import AutoModelForCausalLM, AutoTokenizer
import traceback

from .attention_maps import compute_attention_maps
from .scenarios import REGISTRY
from .core.utils import (
    hidden_states_reshape,
    attentions_reshape,
    attention_outputs_reshape,
)
from src.metrics.token_uq import (
    logits_uq,
    layer_evolution_uq,
    full_layer_shannon_entropy,
)

import argparse
import torch
import time
import gc
import os


def run(
    dataset_name: str,
    model_name: str,
    suite: str,
    result_path: str,
    temperature: float = 0.5,
    max_length: int = 1024,
    store_hidden_states: bool = False,
    store_attentions: bool = False,
    store_attention_outputs: bool = False,
    store_attention_influence: bool = False,
    store_metrics: bool = False,
    device: str = "cuda:0",
    limit: int | None = None,
):

    file_path = f"{result_path}/{suite}"
    os.makedirs(file_path, exist_ok=True)

    def store_results(results):
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f"{file_path}/{dataset_name}_{model_name.replace('/', '_')}_{time_stamp}.pt"
        torch.save(results, output_file)

    ScenarioClass = REGISTRY[dataset_name]
    scenario = ScenarioClass()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        trust_remote_code=False,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
    )

    # Add hook to store attention outputs
    attention_outputs = {}

    def save_attention_output(layer_idx):
        def hook(module, input, output):
            # Store attention output for this generation step
            if f"layer_{layer_idx}" not in attention_outputs:
                attention_outputs[f"layer_{layer_idx}"] = []
            attention_outputs[f"layer_{layer_idx}"].append(
                output[0].detach().cpu()
            )

        return hook

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.register_forward_hook(save_attention_output(i))

    # Processes one item at a time for simplicity
    results = []
    amount_processed = 0
    while scenario.has_next() and (limit is None or amount_processed < limit):
        print(f"Processing sample {amount_processed + 1}...")

        try:
            sample = scenario.sample()
            if sample is None:
                break

            prompt = sample["prompt"]
            reference = sample["reference"]

            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # Adds proper assistant prompt
                return_tensors="pt",
                return_dict=True,  # Returns dict with input_ids AND attention_mask
            ).to(device=device)

            prompt_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_hidden_states=store_hidden_states or store_metrics,
                    output_attentions=store_attentions
                    or store_attention_influence,
                )

            decoded_sequence = tokenizer.decode(
                outputs.sequences[0, :],
                skip_special_tokens=True,
            )

            result = {
                "prompt": prompt,
                "prompt_length": prompt_length,
                "reference": reference,
                "generation": decoded_sequence,
                "sequences": outputs.sequences.cpu(),
            }

            store_tensors = (
                store_hidden_states or store_attentions or store_metrics
            )

            if store_tensors:
                hidden_states = hidden_states_reshape(
                    outputs.hidden_states
                ).to(device, torch.bfloat16)
                attentions = attentions_reshape(outputs.attentions).to(
                    device, torch.bfloat16
                )
                attentions_outputs = attention_outputs_reshape(
                    attention_outputs
                ).to(device, torch.bfloat16)

                if store_attention_influence:
                    attention_maps = compute_attention_maps(
                        attentions,
                        hidden_states,
                        attentions_outputs,
                    )
                    result["attention_maps"] = attention_maps

                if store_metrics:
                    shannon_uq = logits_uq(
                        hidden_states,
                        model.lm_head,
                        outputs.sequences,
                        "logits_shannon_entropy",
                    )

                    full_layer_shannon_uq = full_layer_shannon_entropy(
                        hidden_states, model.lm_head
                    )

                    nll = logits_uq(
                        hidden_states,
                        model.lm_head,
                        outputs.sequences,
                        "logits_negative_log_likelihood",
                    )

                    layer_shannon_variance = layer_evolution_uq(
                        hidden_states,
                        model.lm_head,
                        "layer_evolution_var_shannon_entropy",
                        layers_from_end=5,
                    )

                    result["layer_shannon_entropy_variance"] = (
                        layer_shannon_variance.cpu()
                    )
                    result["token_shannon_entropy"] = shannon_uq.cpu()
                    result["full_layer_shannon_entropy"] = (
                        full_layer_shannon_uq.cpu()
                    )
                    result["token_nll"] = nll.cpu()

                if store_hidden_states:
                    result["hidden_states"] = hidden_states.cpu()

                if store_attentions:
                    result["attentions"] = attentions.cpu()

                if store_attention_outputs:
                    result["attention_outputs"] = attentions_outputs.cpu()

                del hidden_states, attentions, attentions_outputs

            attention_outputs = {}
            torch.cuda.empty_cache()
            gc.collect()

            results.append(result)
            amount_processed += 1

            # Progress indicator
            if amount_processed % 5 == 0:
                print(f"Processed {amount_processed} samples...")

                store_results(results)
                results = []
                gc.collect()

        except Exception as e:
            print(f"Error processing sample {amount_processed + 1}: {str(e)}")
            print(traceback.format_exc())
            store_results(results)
            results = []
            gc.collect()

            continue

    if results:
        store_results(results)

    print(f"Benchmark completed. Results saved to {file_path}")
    print(f"Total samples processed: {amount_processed}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to use (must be registered in REGISTRY)",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the model to evaluate",
    )

    parser.add_argument(
        "--suite", type=str, required=True, help="Name of the evaluation suite"
    )

    parser.add_argument(
        "--result_path", type=str, required=True, help="Path to save results"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for text generation",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum length for generated text",
    )

    parser.add_argument(
        "--store_hidden_states",
        type=bool,
        default=False,
        help="Whether to store hidden states (True/False)",
    )

    parser.add_argument(
        "--store_attentions",
        type=bool,
        default=False,
        help="Whether to store attention weights and outputs (True/False)",
    )

    parser.add_argument(
        "--store_metrics",
        type=bool,
        default=False,
        help="Whether to store the best performing metrics",
    )

    parser.add_argument(
        "--store_attention_outputs",
        type=bool,
        default=False,
        help="Whether to store attention outputs (True/False)",
    )

    parser.add_argument(
        "--store_attention_influence",
        type=bool,
        default=False,
        help="Whether to compute and store attention influence metrics (True/False)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to process (useful for testing)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on (e.g., 'cuda:0', 'cpu')",
    )

    return parser.parse_args()


def main():
    """Main entry point for command line execution."""
    try:
        args = parse_arguments()

        # Validate dataset name
        if args.dataset_name not in REGISTRY:
            available_datasets = list(REGISTRY.keys())
            raise ValueError(
                f"Dataset '{args.dataset_name}' not found in registry. "
                f"Available datasets: {available_datasets}"
            )

        # Validate device
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            print(
                "Warning: CUDA requested but not available. Falling back to CPU."
            )
            args.device = "cpu"

        print("\nStarting benchmark with the following configuration:\n")
        print(f"  Dataset: {args.dataset_name}")
        print(f"  Model: {args.model_name}")
        print(f"  Suite: {args.suite}")
        print(f"  Device: {args.device}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Max length: {args.max_length}")
        print(f"  Limit: {args.limit if args.limit else 'No limit'}")
        print(f"  Results will be saved to: {args.result_path}")
        print("-" * 30)

        run(
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            suite=args.suite,
            result_path=args.result_path,
            temperature=args.temperature,
            max_length=args.max_length,
            device=args.device,
            limit=args.limit,
            store_attentions=args.store_attentions,
            store_attention_outputs=args.store_attention_outputs,
            store_attention_influence=args.store_attention_influence,
            store_hidden_states=args.store_hidden_states,
            store_metrics=args.store_metrics,
        )

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        raise


if __name__ == "__main__":
    main()
