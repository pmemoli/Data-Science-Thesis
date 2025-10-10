from .scenario import Scenario
import datasets
import random


class MATH(Scenario):
    def __init__(self):
        super().__init__()
        self.load_dataset()

    def load_dataset(self):
        print("Loading MATH dataset...")

        dataset = datasets.load_dataset("nlile/hendrycks-MATH-benchmark")
        for item in list(dataset["test"]):
            question = item["problem"].strip()
            reference = item["solution"].strip()

            self.items.append({"question": question, "reference": reference})

        random.shuffle(self.items)
