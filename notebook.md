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

## July 12th 2025:

Today i'm starting to experiment with different entropy metrics based exclusively on the token output probabilities. I have 4 metrics to test:

### Predictive entropy:

This is defined as the normalized surprisal of the generated sequence. Normalization is done by dividing the entropy by the sequence length.

$$PE(S, X) = -log(P(S|X)) / N = -\sum_i log(P(S_i | S_{< i, x})) / N$$

The definition comes from the Kadvath (2022) paper.

### Shannon entropy:

This is the average shannon entropy of the next token probabilities for each token in the sequence. I haven't seen this anywhere, but it feels like a natural metric to compute.

### More sofisticated metrics:

- Semantic entropy: https://www.nature.com/articles/s41586-024-07421-0
- Word-Sequence Entropy: https://arxiv.org/html/2402.14259v1

Today I practiced some pytorch and implemented the first 2 since they are super simple to compute. I also wrote a simple function to evaluate a llms performance on the gsm8k dataset, while computing all the relevant metrics. The function is pretty modular, so extending it to more metrics or models should be easy.

The next steps are finishing the details of that function and trying it on one of Lucianos GPUS.

## July 14th 2025:

I finished the analizer function to test a bunch of metrics on the gsm8k dataset, and debugged some issues with the metrics on batches. The next step is to actually run it on a GPU and compute the AUROC and whatever for the different metrics.

## July 15th 2025:

I have everything ready to run the experiments on the gsm8k dataset. But I had trouble with actually connecting to a GPU. It's probably a good idea to spend some time learning how to ssh to a machine and run stuff on it. I also wasn't able to get into the machines with teamviewer...

## July 16th 2025:

Still can't connect! But to make the most of the time, I begun thinking about possible metrics and functionalities that are going to be useful for the thesis. One thing we talked about with my director there may be merit into estimating the distributions, but with the embeddings from the different attention blocks, rather than just the last one.

I played a bit with the hidden states from each attention layer, computing the probabilities based on the corresponding logits. I imagine that maybe I can get a more sophisticated measure of uncertainty by working with how the probabilities change across the layers...

I extended the inference funtion to compute the probabilities distribution of [batch_size, layers, sequence_length, vocab_size]. The metrics have to be adapted to work with this new shape, but it should be super easy. This provides me a LOT of data to experiment with different metrics.

## July 17th 2025:

I finally was able to ssh into the GPU machine! But there are still tinynthings to make the development experience seamless. Nevertheless, good progress!

## July 18th 2025:

Today I'm finally running the experiments on the gsm8k dataset.

After debugging some issues relating to huge memory sizes form the inference outputs, I was able to compute the AUROC for 20 elements of the gsm8k dataset (not really representative, but I just wanted to test the code). I got the following results:

Predictive Entropy AUROC: 0.7467
Shannon Entropy AUROC: 0.7667

Which is very promising! Nevertheless I need to run on more elements to get an actual idea of the performance.

The next steps are running this on like 1000 elements, and trying some other metrics.

## July 19th 2025:

I made the inference function much more memory efficient, especially when not the hidden state outputs are not required. The previous implementation ran out of memory very frecuently.

The results on the first 1000 elements of the gsm8k dataset (train) are:

Predictive Entropy AUROC: 0.7400
Shannon Entropy AUROC: 0.7528

The results are pretty good! The next step is trying some other metrics. Word-Sequence Entropy is the next one on the list: https://arxiv.org/html/2402.14259v1#bib.bib19

## July 23th 2025:

Took a break to focus on work and exams. Didn't do much today either, but I found some really cool papers that I want to read:

- https://www.researchgate.net/publication/389315584_Entropy-Lens_The_Information_Signature_of_Transformer_Computations

Shows how the entropy distribution of the logits can be used to understand the model's behavior. Nice results and its very relevant!.

- https://arxiv.org/pdf/2410.22685

Provides a computationally efficient way to compute uncertainty, while avoiding the need to sample from the model. The only thing is that it requires fine-tuning the model according to some output distribution. It also compares the AUROC of PE entropy against their version, which is exactly the same thing i am doing!

I think its probably a good idea to read these papers and see if I can get a cool method from them. I am in the mood of trying a bunch of different metrics (from papers and invented) and see what works best. From the second paper, it appears that the best result is going to come from training of some sort.

## July 25th 2025:

Today I'm reading "Entropy-Lens: The Information Signature of Transformer Computations" to get a better intuition about the entropy based metrics.

---

They analize the evolutoin of the shannon entropy of the generated tokens after each intermediate block in the residual stream (output of attention layers?). Based on this they find that:

- The evolution identifies the model family
- Identifies the task type
- Is correlated with the model's performance

They define the "entropy profile" as a matrix $M$ where $M_{i,j}$ is the entropy of the $i$-th token after the $j$-th block $H_i^j$ (mimics form of the transformer horizontally). After that they train a KNN to classify the task type and accuracy for multiple choice tasks, based on the entropy profile.

I imagine they use a fix amount of output tokens to make the matrix a fixed size, possibly padding with 0 entropy.

They also find that the average entropy within a block has a unique geometry when plotted in a graph (avg entropy vs layer). That is super interesting! especially since the form is identifiable regardless of the model size.

One really cool observation is that entropy is highest in the middle layers. They hypothesize that this is because the model is "exploring" the space of possible outputs, and then "converging" to a more certain output in the last layers.

---

Overall I found the paper pretty badly written and pretentious with notation. Nevertheless its interesting! I also already wrote the code to compute the entropy profile. It remains to be seen what I can do with it...

One thing is certain, I WANT a metric based on the entire entropy profile, not just the output of the last layer. I also think that there may be some merit in training something on that profile. Just computing a single number with an average or whatever without training anything seems overly simplistic.

Since that took little time i'm also going to read the second paper I mentioned on the 23rd.

- IMPROVING UNCERTAINTY QUANTIFICATION IN LARGE LANGUAGE MODELS VIA SEMANTIC EMBEDDINGS

The paper proposes a novel method to measure uncertainty based on the embeddings, rather than the sequence likelihood and bi-directional entailment criteria (see later what that is). The results show a considerable improvement in the AUROC compared to the PE and semantic similarity entropy. Also, they propose a computationally efficient version!

Intruction:

Raw entropy metrics mix the uncertainty from sintax and semantics, making the metrics less reliable to measure uncertainty of meaning. Semantic entropy is a distinct measure which aims to isolate sematnic uncertainty. The issue is that they require sampling from an input by varying the seed, which is computationally expensive.

They introduce:

- Semantic Embedding Uncertainty (SEU): Levarage average pairwise cosine similarity (sampling many outputs???)

- Amortized Semantic Embedding Uncertainty (ASEU): A computationally efficient version of SEU that models semantics as latent variables (requires training?).

## Semantic Embedding Uncertainty (SEU):

They sample $N$ output sequences and obtain an embedding (from another model) for each sequence. After that they compute the pairwise cosine similarity between the embeddings and define the SEU as:

SEU(x) = 1 - 2 / M(M-1) \* \sum*{i=1}^{M} \sum*{j=1}^{M} cos(emb_i, emb_j)

Where $M$ is the number of sampled sequences.

They experiment wiht M = 5 and temperature as 0.5 (from a previour work on SE) on small models like Phi-3.5-instruct. They use AUROC as a metric to measure performance. SEU greatly outperforms PE and SE, with an AUROC of like 0.85 on the TRIVIAQA dataset! Where as PE scores like 0.5-65.

## Amortized Semantic Embedding Uncertainty (ASEU):

Kinda tired. On monday I'm continuint reading about ASEU.
