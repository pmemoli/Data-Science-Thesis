from .scenario import Scenario
import datasets
import random


class GSM8K(Scenario):
    def __init__(self):
        super().__init__()
        self.load_dataset()

    def load_dataset(self):
        print("Loading GSM8K dataset...")

        dataset = datasets.load_dataset("gsm8k", "main")
        for item in list(dataset["test"]):
            question = item["question"].strip()
            reference = item["answer"].strip()

            self.items.append({"question": question, "reference": reference})

        random.shuffle(self.items)
