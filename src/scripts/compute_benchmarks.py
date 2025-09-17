# %%
from litellm import completion
from pydantic import BaseModel
import json
import time
import torch
import os

# %%
suite = "validation"
data_path = f"src/data/runs/{suite}"
filename = "gsm8k_microsoft_Phi-3.5-mini-instruct_20250916-210355.pt"


# %%
class Response(BaseModel):
    success: bool


api_key = os.getenv("GEMINI_API_KEY")  # type: ignore


def evaluate_response(
    question: str, response: str, correct_answer: str
) -> bool:
    # Clean response
    cleaned_response = (
        response.replace("<|end|>", "").replace("<|endoftext|>", "").strip()
    )

    # Quick normalization check first

    agent_prompt = f"""
Question: {question}

Response: {cleaned_response}

Correct Answer:

{correct_answer}
"""

    messages = [
        {
            "role": "system",
            "content": """Task: Determine if the response is corresponds to the correct answer for the question, based on the given Correct Answer text. 

            Answer ONLY with the exact format: {{"success": True}} or {{"success": False}}""",
        },
        {"role": "user", "content": agent_prompt},
    ]

    result = completion(
        model="gemini/gemini-2.5-flash",
        messages=messages,
        api_key=api_key,  # type: ignore
    )

    json_content = str(result.choices[0].message.content)  # type: ignore

    return "true" in json_content.lower()


# %%
data = torch.load(
    os.path.join(data_path, filename), map_location="cpu", mmap=True
)

# %%
import hashlib


def hash_result(question, response):
    hasher = hashlib.sha256()
    hasher.update(question.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(response.encode("utf-8"))
    return hasher.hexdigest()


# %%
evaluations = {}

# %%
for item_idx in range(len(data)):
    for i in range(4):
        try:
            item = data[item_idx]

            question = item["prompt"]
            response = item["generation"]
            correct_answer = item["reference"]

            success = evaluate_response(question, response, correct_answer)

            evaluations[hash_result(question, response)] = success

            print("\n")
            print(f"Processing item {item_idx + 1}/{len(data)}...")
            print(f"Success: {success}")

            break

        except Exception as e:
            time.sleep(2**i)  # Exponential backoff

# %%
# Save evaluations
with open(
    os.path.join(data_path, f"evaluations_{filename.replace('.pt', '.json')}"),
    "w",
) as f:
    json.dump(evaluations, f)
