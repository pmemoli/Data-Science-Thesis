from .gsm import GSM8K
from .math import MATH
from .gpqa import GPQA
from .scibench import SCIBENCH

REGISTRY = {"gsm8k": GSM8K, "math": MATH, "gpqa": GPQA, "scibench": SCIBENCH}
