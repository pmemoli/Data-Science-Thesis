# Informal notebook to test stuff

# %%
from pydantic import BaseModel
from litellm import completion
import argparse
import torch
import os

api_key = os.getenv("GEMINI_API_KEY")  # type: ignore


class Response(BaseModel):
    success: bool


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
suite = "gsm-test"
tensor_path = f"src/data/runs/{suite}"
tensor_files = os.listdir(tensor_path)

for file in tensor_files[2:]:
    full_path = f"{tensor_path}/{file}"
    tensor = torch.load(full_path)
    for tensor_item in tensor:

        success = evaluate_response(
            tensor_item["prompt"],
            tensor_item["generation"],
            tensor_item["reference"],
        )

        tensor_item["success"] = success

        break

    torch.save(tensor, full_path)

    break
