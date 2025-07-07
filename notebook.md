# Notebook

I'm tracking the daily progress and observations of the thesis in this markdown notebook.

## July 7th 2025:

**Objective**: The (current) goal of the thesis is to find a task-agnostic metric for the performance of LLMs on a domain. The metric should be exclusively a function of the model's state, and not depend on the task or the dataset.

A friend shared a paper that is quite relevant. It is titled "Detecting hallucinations in large language models using semantic entropy" and was published in Nature.

The paper discusses a method for detecting a specific type of "confabulations" hallucinations in LLMs. These are hallucinations that are wrong and arbitrary, by which they mean that the answer is sensitive to irrelevant details such as a random seed. They distinguish this from "misinformation" hallucinations, which are wrong but not arbitrary and are a result of being trained on erroneous data.

**Doubt**: What does "random seed" mean in this context? Does it refear to the random seed with which the next token is sampled according to some temperature?

One immediate observation is that "misinformation" hallucinations are probably going to be very hard to detect with a metric that is purely a function of the model's state. I imagine that the model's state is not going to be very different when it generates a "misinformation" errors compared to when it generates a correct answer. It remains to be seen if these "misinformation" errors are one of the main sources of poor LLM performance, or if they just dissapear with scale (more training data).

The fundamental idea is that when the model is generating a confabulation, it will have a high semantic entropy. This leads to next-token sampling being both:

1. More sensitive to irrelevant details
2. Much less deterministic given some degree of temperature

The method proposed in the paper is to use a measure of "semantic entropy" based on the next token probabilities. They distinguish between "semantic entropy", and "naive entropy", which is simply the entropy of the next token probabilities over a sequence. This is important, since different sequences of text can mean exactly the same thing, and the naive entropy would be high for these sequences, even if they are semantically equivalent.

### Semantic Entropy

### AUROC metric

### Results
