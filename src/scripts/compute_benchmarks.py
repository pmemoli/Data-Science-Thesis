# Evaluates

# %%
from litellm import completion
from pydantic import BaseModel
import json
import os

# %%
model = "microsoft_phi-3.5-mini-instruct"
suite = "helm-lite-custom"
data_path = f"src/data/helm/runs/{suite}"

by_scenario = {
    "narrative_qa": [f"narrative_qa:model={model},max_train_instances=0"],
    "mmlu": [
        f"mmlu:subject=abstract_algebra,method=multiple_choice_joint,model={model},max_train_instances=0",
        f"mmlu:subject=college_chemistry,method=multiple_choice_joint,model={model},max_train_instances=0",
        f"mmlu:subject=computer_security,method=multiple_choice_joint,model={model},max_train_instances=0",
        f"mmlu:subject=econometrics,method=multiple_choice_joint,model={model},max_train_instances=0",
        f"mmlu:subject=us_foreign_policy,method=multiple_choice_joint,model={model},max_train_instances=0",
    ],
    "math": [
        f"math:subject=number_theory,level=1,use_official_examples=False,use_chain_of_thought=True,model={model},max_train_instances=0",
        f"math:subject=intermediate_algebra,level=1,use_official_examples=False,use_chain_of_thought=True,model={model},max_train_instances=0",
        f"math:subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True,model={model},max_train_instances=0",
        f"math:subject=prealgebra,level=1,use_official_examples=False,use_chain_of_thought=True,model={model},max_train_instances=0",
        f"math:subject=geometry,level=1,use_official_examples=False,use_chain_of_thought=True,model={model},max_train_instances=0",
        f"math:subject=counting_and_probability,level=1,use_official_examples=False,use_chain_of_thought=True,model={model},max_train_instances=0",
        f"math:subject=precalculus,level=1,use_official_examples=False,use_chain_of_thought=True,model={model},max_train_instances=0",
    ],
    "gsm": [f"gsm:model={model},stop=none,max_train_instances=0"],
    "legalbench": [
        f"legalbench:subset=abercrombie,model={model},max_train_instances=0",
        f"legalbench:subset=corporate_lobbying,model={model},max_train_instances=0",
        f"legalbench:subset=international_citizenship_questions,model={model},max_train_instances=0",
        f"legalbench:subset=function_of_decision_section,model={model},max_train_instances=0",
        f"legalbench:subset=proa,model={model},max_train_instances=0",
    ],
    "med_qa": [f"med_qa:model={model},max_train_instances=0"],
}


# %%
class Response(BaseModel):
    success: bool


api_key = os.getenv("GEMINI_API_KEY")  # type: ignore


def normalize_text(text: str) -> str:
    import re

    text = text.lower().strip()
    text = re.sub(r"\b(the|a|an)\s+", "", text)  # Remove articles
    text = re.sub(r"'s?\b", "", text)  # Remove possessives
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text


def evaluate_response(question: str, response: str, correct_answers: list[str]) -> bool:
    # Clean response
    cleaned_response = (
        response.replace("<|end|>", "").replace("<|endoftext|>", "").strip()
    )

    # Quick normalization check first
    normalized_response = normalize_text(cleaned_response)
    for correct_answer in correct_answers:
        if normalized_response == normalize_text(correct_answer):
            return True

    answer_str = "\n".join([f"- {ans}" for ans in correct_answers])

    agent_prompt = f"""
Question: {question}

Response: {response}

Correct Answers:

{answer_str}
"""

    messages = [
        {
            "role": "system",
            "content": """Task: Determine if the response is equivalent to any of the correct answers. 

            Consider these as correct matches:
            - Exact matches (ignoring case and punctuation)
            - Semantically equivalent answers (same meaning, different wording)
            - Answers that contain the essential correct information

            Note that for multiple choice answers, the response must match the correct one despite any additional explanation.

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
values = []

for scenario, runs in by_scenario.items():
    if scenario == "narrative_qa":
        continue

    for run in runs:
        print(f"Processing {run} in scenario {scenario}...\n")

        path = f"{data_path}/{run}/scenario_state.json"
        with open(path, "r") as f:
            scenario_state = json.load(f)

        instances = scenario_state["request_states"]

        for instance in instances:
            prompt = instance["request"]["prompt"]
            response = instance["result"]["completions"][0]["text"]

            references = instance["instance"]["references"]

            correct_response = references[0]["output"]["text"]

            correct_answers = []
            for reference in references:
                if "correct" in reference["tags"]:
                    correct_answers.append(reference["output"]["text"])

            eval_result = evaluate_response(
                question=prompt, response=response, correct_answers=correct_answers
            )

            instance["evaluation"] = eval_result

            print(
                answer_str := f"Q: {prompt}\nA: {response}\nCorrect: {correct_answers}\nEval: {eval_result}\n"
            )

        with open(path, "w") as f:
            json.dump(scenario_state, f, indent=2)
