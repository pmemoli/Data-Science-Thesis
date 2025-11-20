import torch
import os

suite = "gsm-test"
tensor_path = f"src/data/runs/{suite}"
tensor_files = os.listdir(tensor_path)

positive_agregate_stats = {}
negative_agregate_stats = {}

i = 0
for file in tensor_files:
    full_path = f"{tensor_path}/{file}"
    tensor = torch.load(full_path)
    for tensor_item in tensor:
        print(
            tensor_item["attention_maps"]["additive_mean_max_last_token"].shape
        )
        print(tensor_item["prompt_length"])
        print(tensor_item["sequences"].shape)

        print(tensor_item["attention_maps"]["influence_projection__max_mean"])

        break

    break
