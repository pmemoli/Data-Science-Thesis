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

## July 8th 2025:

Today I'm reviewing the evaluation metrics and results from the paper. For this I'm reviewing the fundamental metrics:

### Evaluation Metrics

Our variable is whether the sequence has a confabulated hallucination or not.

#### ROC CURVE:

The ROC curve is a graphical representation of how well a binary classifier which outputs some measure of certainty performs.

WLOG the certainty measure is a probability measure.

The roc curve plots the:

Y: TRUE POSITIVE RATE (or recall, the proportion of POSITIVE cases that were identified as positive by the classifier)
against the
X: FALSE POSITIVE RATE (proportion of NEGATIVE cases that were identified as positive by the classifier)

Each of which is estimated on a dataset.

The classification is performed by comparing the certainty measure against some threhsold:

Classifier(x) = 1(P(Y=1 | x) > threhold)

So, for a small threshold, the recall is very high, but so is the false positive rate. Thus TPR(FPR(threhsold)) is monotonous for every composition.

AUROC is simply the area under this curve.

- A perfect classifier sets P(Y=1 | x) as 1 or 0, so the threshold becomes irrelevant (assuming it is between 0 and 1). Thus the AUROC is 1.

- A perfectly random classifier has on average an auroc of 0.5

- A bad classifier has an auroc of < 0.5, and > 0.5 if its good

They use other metrics that tweak the AUROC, but i just don't think it is relevant spending more time on this.

### Assesing accuracy

Just like yesterday, i'm ignoring the details of the short sequence accuracy estimation. For long sequences, they decide whether the sequence is a confabulation or not through a GPT 4 prompt that decides if the proposed answer to a question is the same as the expected:

We are assessing the quality of answers to the following question:
{question}
The expected answer is: {reference answer}
The proposed answer is: {predicted answer}
Within the context of the question, does the proposed answer mean
the same as the expected answer? Respond only with yes or no.

### Results

They found an AUROC of ~0.75 with semantic entropy for detecting confabulations, whereas self-checking corresponds to an AUROC of ~0.5.

### Takaways and possible next steps

The results are very promising for the thesis! They indicate that it is indeed possible to estimate the performance just as a function of the model logits.

The paper provides a metric that serves as a baseline for the thesis objective. Even if their aim was "detecting confabulation hallucinations", they are really just comparing the "uncertainty" of the generated sequence against the correctness of the response.

A possible first step is to simply to choose a domain, and use this semantic entropy metric as a measure of performance by computing the corresponding AUROC. I can possibly tweak and play a bit with the algorithm that computes semantic entropy.

## July 9th 2025:

I talked with Luciano about the next steps. We agreed on the following:

1. Use phi3 mini instruct as an initial benchmark: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
2. Choose two datasets -one where the model is expected to perform well, and one where it is expected to perform poorly- from the paper: https://arxiv.org/pdf/2404.14219
3. Use the semantic entropy metric from the nature paper to compute the AUROC for both datasets

The semantic entropy serves as a good baseline upon which I can build. Today I'm just going to download the model and the datasets (1 and 2), and get familiar with the relevant libraries.

I think i'm sticking math datasets as a first approach. The accuracy evaluation is much easier, and I have two similar datasets that can be compared.

- GSM8K: Performs quite well
- MATH: Performs poorly

## July 10th 2025:

I'm continuing developing the code to compute different metrics in a modular way. The idea is to be able to easily plug in different metrics and datasets, and compare them easily.

The transformers library is SO bad and undocumented. Nevertheless I was able to write a cute inference function that returns the sequence probabilities and generated text. From that I can calculate a bunch of entropy based metrics.
