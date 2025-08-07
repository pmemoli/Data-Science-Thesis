from dataclasses import dataclass
from typing import Literal
from torch import Tensor

Metric = Literal["predictive_entropy", "shannon_entropy", "attention_entropy"]


@dataclass
class DatasetResult:
    indexes: list[int]
    incorrect_answers: list[bool]
    metrics: dict[Metric, list[float]]
    model_answers: list[str]
    correct_answers: list[str]


@dataclass
class InferenceOutput:
    # [batch_size][sequence_length]
    generated_ids: Tensor
    token_probabilities: Tensor

    # [batch_size][layers][sequence_length][top_k] (layers can be just the last one)
    token_distribution: Tensor
    token_distribution_ids: Tensor

    # [batch_size]
    generated_text: list[str]
    sequence_length: Tensor

    # list([batch_size][heads][seq_length])
    attentions: list[Tensor] | None = None
