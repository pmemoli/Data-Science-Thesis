from transformers import pipeline

from .scenarios import REGISTRY
from .metrics.white_box import early_exit_uq, last_layer_distribution_uq, layer_evolution_uq, metrics, aggregation_methods, weighting_methods

import argparse
import torch
import json
import os 

def run_benchmark(
    dataset_name: str,
    model_name: str,
    suite: str,
    result_path: str,
    temperature: float = 0.5,
    max_length: int = 1024,
    batch_size: int = 1,
    sample_amount: int = 1,
    device: str = "cuda:0",
    limit: int | None = None,
):
    ScenarioClass = REGISTRY[dataset_name]
    scenario = ScenarioClass()
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=False,
        device=device,
    )

    tokenizer = pipe.tokenizer
    model = pipe.model
    
    # Processes one item at a time for simplicity
    results = []
    amount_processed = 0
    while scenario.has_next() and (limit is None or amount_processed < limit):
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
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states
        sequences = outputs.sequences[:, inputs.input_ids.shape[1]:]
        lm_head = model.lm_head

        # Compute metrics
        grid = {metric: {} for metric in metrics}

        for metric in metrics:
            for agg in aggregation_methods:
                pooling_ratio = 1.0

                if agg == "5%_sequence":
                    pooling_ratio = 0.05
                elif agg == "10%_sequence":
                    pooling_ratio = 0.1
                elif agg == "20%_sequence":
                    pooling_ratio = 0.2

                for weight in weighting_methods:
                    if "early_exit" in metric:
                        metric_name = metric.replace("early_exit_", "")

                        for threshold in [0.025, 0.05, 0.1, 0.2]:
                            score = early_exit_uq(
                                metric_name=metric_name, # type: ignore
                                hidden_states=hidden_states,
                                lm_head=lm_head,
                                threshold=threshold,
                                sequences=sequences,
                                weighting=weight,
                                pooling_ratio=pooling_ratio,
                                pad_token_id=pipe.tokenizer.pad_token_id,
                            )

                            grid[metric][f"{agg}_{weight}_{threshold}"] = score

                    elif "last_layer_distribution" in metric:
                        metric_name = metric.replace("last_layer_distribution_", "")

                        score = last_layer_distribution_uq(
                            metric_name=metric_name, # type: ignore
                            hidden_states=hidden_states,
                            lm_head=lm_head,
                            sequences=sequences,
                            weighting=weight,
                            pooling_ratio=pooling_ratio,
                            pad_token_id=pipe.tokenizer.pad_token_id,
                        )

                        grid[metric][f"{agg}_{weight}"] = score

                    elif "layer_evolution" in metric:
                        metric_name = metric.replace("layer_evolution_", "")

                        score = layer_evolution_uq(
                            metric_name=metric_name, # type: ignore
                            hidden_states=hidden_states,
                            lm_head=lm_head,
                            sequences=sequences,
                            weighting=weight,
                            pooling_ratio=pooling_ratio,
                            pad_token_id=pipe.tokenizer.pad_token_id,
                        )

                        grid[metric][f"{agg}_{weight}"] = score

                    else:
                        continue

        result = {
            "prompt": prompt,
            "reference": reference,
            "generation": tokenizer.decode(
                sequences[0],  # Fixed: decode individual sequence
                skip_special_tokens=True,
            ),
            "metrics": grid,
        }

        results.append(result)
        amount_processed += 1
        
        # Progress indicator
        if amount_processed % 10 == 0:
            print(f"Processed {amount_processed} samples...")

    # Store results
    file_path = f"{result_path}/{suite}"
    os.makedirs(file_path, exist_ok=True)

    output_file = f"{file_path}/{dataset_name}_{model_name.replace('/', '_')}.jsonl"
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")  # Fixed: proper JSON formatting
    
    print(f"Benchmark completed. Results saved to: {output_file}")
    print(f"Total samples processed: {amount_processed}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to use (must be registered in REGISTRY)"
    )
    
    parser.add_argument(
        "--model_name", 
        type=str,
        required=True,
        help="Name or path of the model to evaluate"
    )
    
    parser.add_argument(
        "--suite",
        type=str, 
        required=True,
        help="Name of the evaluation suite"
    )
    
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path to save results"
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for text generation"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum length for generated text"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int,
        default=1,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "--sample_amount",
        type=int,
        default=1,
        help="Number of samples to generate per prompt"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on (e.g., 'cuda:0', 'cpu')"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to process (useful for testing)"
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
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            args.device = "cpu"
        
        print("Starting benchmark with the following configuration:")
        print(f"  Dataset: {args.dataset_name}")
        print(f"  Model: {args.model_name}")
        print(f"  Suite: {args.suite}")
        print(f"  Device: {args.device}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Max length: {args.max_length}")
        print(f"  Limit: {args.limit if args.limit else 'No limit'}")
        print(f"  Results will be saved to: {args.result_path}")
        print("-" * 50)
        
        run_benchmark(
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            suite=args.suite,
            result_path=args.result_path,
            temperature=args.temperature,
            max_length=args.max_length,
            batch_size=args.batch_size,
            sample_amount=args.sample_amount,
            device=args.device,
            limit=args.limit
        )
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        raise


if __name__ == "__main__":
    main()
