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

I summarize my understanding of the following sections:

### Entropy

Uncertainty of the output distribution is measured through predictive entropy given a specific input $x$. This consists of the entropy of the output conditioned by the input:

$PE(x) = H(Y | x) = - \sum_{y \in rg(Y)} P(y|x) * ln(P(y|x))$

#### Joint Probability of sequence of tokens

Given a generated sequence $s$, the corresponding log probability of the sequence according to the LLM is:

$log(P(s | x)) = \sum_i log(P_i|s_{<i}, x)$

When comparing log probabilities, the authors normalize the probabilities through a division by the sequence length.

### Semantic Entropy

The idea is that the uncertainty of the token distribution is not necessarily the same as the uncertainty of its meaning. Even if the model is quite sure of its response, there can be many ways of saying the same thing.

The semantic entropy metric seeks to estimate the uncertainty of the meaning of its generation, not just the choice of words. This involves 3 steps:

1. Sample output sequences of tokens from the predictive distribution of an LLM
2. Cluster sequences by meaning using some algorithm (they use another language model)
3. Compute the cluster entropy

### Evaluation Metrics

The authors use AUROC and AURAC, where the "accuracy" of the response is measured through:

LONG SEQUENCES:

They input this into gpt 4

```
{question}
The expected answer is: {reference answer}
The proposed answer is: {predicted answer}
Within the context of the question, does the proposed answer mean
the same as the expected answer? Respond only with yes or no
```

For short sequences they use some other method I don't quite understand

### Results

Too tired, I'm reviewing the paper again tomorrow.
