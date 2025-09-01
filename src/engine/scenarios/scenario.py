from ..core.types import DatasetItem, Task, SampleResult
from ..core.prompts import task_prompts

class Scenario:
    def __init__(self):
        self.items: list[DatasetItem] = []

    def load_dataset(self):
        pass # Implemented in subclasses

    def sample(self, format:Task | None=None) -> SampleResult | None:
        dataset_item = self.items.pop() if self.items else None
        if dataset_item is None:
            return None

        question = dataset_item["question"]
        reference = dataset_item["reference"]

        if format is None:
            prompt = question

        else:
            prompt = task_prompts[format]
            prompt = prompt.format(question=question)

        return {"prompt": prompt, "reference": reference}

    def has_samples(self) -> bool:
        return len(self.items) > 0
