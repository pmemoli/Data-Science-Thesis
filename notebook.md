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

## July 28th 2025:

Today i'm continuing reading the ASEU paper. They introduce a latent variable model for this, so it makes sense reviewing basic bayesian inference.

### Bayesian Inference

In bayesian statistics, the parameters are modeled as random variables, so it makes sense conditioning by the parameters:

- Posterior: P(\theta | x) (probability of the parameters given the data)
- Prior: P(\theta) (probability of the parameters before seeing the data)
- Likelihood: P(x | \theta) (probability of the data given the parameters)

They relate through Bayes' theorem:

P(\theta | x) = P(x | \theta) \* P(\theta) / P(x)

And hence the Posterior is proportional to the likelihood \* prior.

Bayesian parameter estimation works by starting from a reasonable prior, and updating it with the data to obtain the posterior. The posterior is then used to make predictions about new data.

### Latent Variable Models

Latent variables are variables that are not directly observed, but are inferred from the data. The joint distribution of the observed data and the latent variables is given by:

- P(x, z) = P(x | z) \* P(z)

Where $x$ is the observed data and $z$ is the latent variable we can't directly observe.

The goal is computing the posterior P(z | x).

### ASEU

Each output sequence of t tokens is modeled as a latent variable $z_t$.

The posterior is approximated as:

P(z_t | x_t, w) ~ N(\mu_t, \sigma_t)

x_t consists of the hidden states of the input sequence plus tokens generated, and \mu_t and \sigma_t are the outputs of a small feed forward network that takes the hidden states as input. Sigma is also assumed to be diagonal for simplicity.

For each value of the output sequence, the model samples a couple of values and computes the pairwise cosine similarity between the latent variables. The ASEU is then defined as:

ASEU = 1 - median{S_1, ..., S_N}

### Results

ASEU is comparable to SEU and considerably better than SE! In the context of the fundamental idea of providing a robust metric for LLM performance, ASEU is very promising! Even if it does require training, it is a neglegible amount of training compared to the model itself.

Cool! I also prompted gemini deepsearch for other relevant papers. These are the ones it suggested:

- The Bayesian Lens on Textual Parameters
- Semantic Volume and the Duality of Uncertainty
- LM-Polygraph framework

The first two seem computationally costly compared to ASEU, and the third one is just a benchmark for measuring uncertainty. I feel toying with ASEU-similar ideas is the way to go.

One thing I notice is that ASEU is simply a function of the models last hidden state, but maybe I could also use the entropy signature from the Entropy-Lens paper to compute a more robust ASEU... Cool!

## August 1st 2025:

I found a really cool paper involving "attention entropy", which is increadibly cheap to compute (unlike shannon entropy, which requires a lot of memory).

https://arxiv.org/pdf/2412.16545

I have barely read it, but one key point is that "attention entropy is heavily correlated with the performance of the model in the lm task".

I can try and analize if attention entropy also provides some sort of "signature" like shannon entropy does. That can heavily enrich the ASEU metric in some way.

No more reading and thinking, its probably a good idea to start coding some of these ideas and see what works and what doesn't.

TODO (pretty long list lol):

- Test how good attention entropy is at predicting performance (just looking at the last layer).

- Train a generative latent variable model just looking at the last hidden state, and using it to compute ASEU.

- Think of how I could enrich the input for the latent variable model with some information signature.

- Probably do some qualitative analysis on how shannon and attention entropy evolve across the layers.

## August 6th 2025:

I made the inference function return the attention values for each attention head on the last layer. I also computed the "attention entropy" in the informal.py script, I have yet to write the corresponding metric function on entropy.py, and test it on gsm8k. Probably tomorrow...

## August 7th 2025:

Had the meeting with my director! Really good meeting. He mentioned the following evaluation framework which can heavily simplify the process https://github.com/EleutherAI/lm-evaluation-harness.

The TODO for the week is to:

- Modify the lm-evaluation-harness to store the hidden states and attention values for each layer, so I can compute the attention entropy and ASEU metrics after evaluation!

- Test on gsm8k and math datasets

- Implement the attention entropy metric

I'm writing a subclass of HFLM from lm-evaluation-harness to store the hidden states and attention values as tensors. After that i'm going to need to merge them somehow.

Basic script to run test evaluations with lm-evaluation-harness:

```
lm_eval --model hf \
    --model_args pretrained=microsoft/Phi-3-mini-4k-instruct,dtype=float16 \
    --apply_chat_template \
    --tasks gsm8k \
    --device cuda:0 \
    --output_path src/data/benchmark_results/gsm8k/gsm8k-phi-3-mini.json \
    --batch_size 1 \
    --log_samples \
    --limit 1
```

I tested it and it works!

For running the subclass, I have to replace the --model hf with my custom model (say state_hf). I still have to figure out how to register the custom model and probably debug it.

Possible fix:

```
lm_eval \
    --model hf \
    --model_args pretrained=gpt2,dtype=float16,trust_remote_code=True \
    --tasks gsm8k \
    --device cuda:0 \
    --output_path src/data/benchmark_results/gsm8k/gpt_2.json \
    --batch_size 1 \
    --log_samples \
    --limit 1
```

And i'd have to register hf-state directly, which is fine honestly.

Good progress today! Tomorrow I'm going to see if I can store the LAST LAYER hidden states and attention values.

## August 8th 2025:

I installed lm-evaluation-harness on the virtual environment. Today i'm going to try and write the custom model.

The computer with the gpus has no memory on the ssd xddd. I just forked the repository and added the custom model on lm_eval/models/huggingface_eval.py. I'm continuing when the memory issue is solved.

## August 11th 2025:

I began heavily modifying the hugginface model directly. To optimize memory usage and inference time, i am modifying the model_generate method directly. Its going to be a good idea to modularize it after it runs correctly.

At night I was able to run the custom hugginface model and it succesfully stores the relevant hidden states and attention values. All thats left is writing the metrics and computing the results (AUROC and average value for each dataset).

## August 12th 2025:

Wrote the code to compute the metrics based on lm-evaluation-harness results! All that is left is running the evaluations on whatever datasets I want, and computing the metrics.

I was thinking of presenting the results in a table, with the following columns:

DATASET, BENCHMARK, PE, PE_AUROC, SE, SE_AUROC, AE, AE_AUROC

I can then just extend the table with more metrics as I compute them!

## August 13th 2025:

Not much to do today. I'm going to make the most of the time and read some fundamental papers. Probably the attention and original gpt paper.

I read and understood in depth these 2 papers:

- Attention is all you need: https://arxiv.org/abs/1706.03762

- Improving language understanding by generative pre-training: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

I'd like to also read in depth the gpt-2, gpt-3 and instructGPT papers. Its crazy how instructGPT came in 2022.

Tomorrow i'm probably reading:

gpt-2 paper: Language Models are Unsupervised Multitask Learners, https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

and (maybe) gpt-3 paper: Language Models are Few-Shot Learners, https://arxiv.org/abs/2005.14165

One key takaway is that language modeling is sometimes enough for actual tasks. And LM is trained on minimizing negative log likelihood (predictive entropy), which is also heavily correlated with attention entropy. This is probably why PE, despite being super simple, works decently well as a performance predictor.

It's going to be interesting to see how rlhf interacts with this minimizing nll objective. I may just be able to extract some better cheap metric.

Late at night I ended up reading the gpt-2 paper. Super interesting! They basically show that scaling the architecture x10 and training on the lm-task leads to the model being able to perform tasks that it was not trained on, such as question answering, summarization, etc. GPT-2 has 1.5 billion parameters.

Current timeline:

Attention is all you need (June 2017) -> GPT-1 (2018) -> GPT-2 (2019)

The gpt 3 paper is looong. I can probably just skim it and go to the instructGPT paper, which is much more important to the thesis.

## August 14th 2025:

My director suggested these papers to read:

Estimating LLM capabilities without labeled data (Findings EMNLP’23)
Predicting performance drop under domain shift w/o target labels (Elsahar et al., 2019)
Uncertainty Estimation & Quantification for LLMs (2024 survey)
Task Calibration (ACL’25 Findings + arXiv’24)
CUE (ACL’25) — harmonized uncertainty scoring that normalizes disparate estimators
Nature (2024) “Larger & more instructable LLMs may be less reliable”
Automated Capability Discovery (NeurIPS’24)

We also fixed the issue with no memory on the ssd. I'm running this for gsm8k and possibly math:

lm_eval --model hf-store \
 --model_args pretrained=microsoft/Phi-3-mini-4k-instruct,dtype=float16 \
 --apply_chat_template \
 --tasks gsm8k \
 --device cuda:0 \
 --output_path src/data/evaluation_results/gsm8k-phi-3-mini.json \
 --batch_size 1 \
 --log_samples \

Got this:

```json
{
  "dataset": "gsm8k",
  "benchmark_score": 0.7664897649734648,
  "se": 0.17119616851029837,
  "se_auroc": 0.6737510758282272,
  "pe": 0.08058323320269675,
  "pe_auroc": 0.6659505183244055,
  "ae": 2.5742901016854263,
  "ae_auroc": 0.5934043701106015
}
```

lm_eval --model hf-store \
 --model_args pretrained=microsoft/Phi-3-mini-4k-instruct,dtype=float16 \
 --apply_chat_template \
 --tasks hendrycks_math \
 --device cuda:0 \
 --output_path src/data/evaluation_results/math-phi-3-mini.json \
 --batch_size 2 \
 --log_samples \

```json
{
  "dataset": "hendrycks_math",
  "se": 0.41608187897354365,
  "pe": 2.4744856816727667,
  "ae": 1.944209218788147
}
```

The SE and PE is muuuch higher for math than for gsm8k. This can be attributed to either the dataset being harder, or the model being worse at MATH, or both. It is 100% worth it computing this results for all the datasets from:

https://huggingface.co/microsoft/Phi-3.5-mini-instruct

And also on a smarter model like the new gpt open source 20b model.

I also realize i've been using phi 3 rather than 3.5.

Today I did this manually, but to do this for all the datasets, I should write a shell script that runs the evaluations and deletes the tensors. I'd have to manually check how they are parsed, and then write the script. Its probably worth it avoiding MATH since its a pain to parse. Doing that is going to also provide me a cool extendable script to test a bunch of metrics.

I also read this paper: https://arxiv.org/pdf/2305.14802 (Estimating LLM capabilities without labeled data) uses NLL as a metric to estimate performance. They grab a dataset and use a "NLL vector" that represents the confidence within the dataset. This NLL vector is feed into a model like xgboost to predict the performance of the model on the dataset. The results look a bit underwhelming, with the average train accuracy sometimes beating the meta models lol.

Its honestly quite relevant to what we are doing. I just don't really like the idea of training a model that directly predicts performance based on the NLL. The features seem to provide a huge intrinsic error, and I feel like training through datasets is not very robust (like no data). I would rather measure something like uncertainty or some other more directly related metric, and based on that estimate performance.

TODO:

1. Choose ~10-15 datasets from the phi 3.5 mini paper
2. Write the proper parser for the results in the compute metric script
3. Write a shell script that runs the evaluations and deletes the tensors
4. Present the results in a cute way

After that:

. Propose a bunch of other metrics (ASEU for sure!)
. Run the evaluations on the gpt 20b model
. Propose some clustering algorithm to find ill-performing domains

## August 18th 2025:

Modifying the loglikelihood method from the library is a huge pain. It is horribly implemented and impossible to understand what each thign does. For this reason I am migrating to standford helm:

https://github.com/stanford-crfm/helm/

Which seems to be much better documented and the code base looks clean. It also already provides the logprobabilities, which is what I want.

Based on the previous results, i am also simply ignoring the attention and shannon entropies. The "naive" metric is the normalized negative log likelihood. ASEU is going to be the "non-naive" metric, which couples nicely with domain clustering since they can use the same embedding model.

TODO: fork helm and compute the NLL on each dataset

helm-run --run-entries mmlu:subject=anatomy,model=openai/gpt2\
 --suite test \
 --output-path src/data/helm/ \
 --max-eval-instances 1

helm-summarize --suite my-suite --o src/data/helm-results/

helm-server --o src/data/helm-results/

It would be a good idea to have a "run metrics" method on the hugginface client that stores the likelihood metrics. It would also be a good idea to add a "metric" parameter to the run command.

Succesfully added a "compute metric" method that stores something in the response! Just have to add PE and SE now (hidden state based metrics in the future, but that is going to be so simple).

## August 19th 2025:

I understand now how multiple choice benchmarks are performed. They feed the question and possible answer to the model, and then just compare the logprobs (normalized ofc). Its not as easy to get metrics ignoring the question like with COT benchmarks, since the hugginface clients don't distinguish between them. I'd have to modify the method which calls the client, and a "question length" parameter to everything...

I'm going to focus on COT benchmarks and only then pass to multiple choice. The context on which this may be useful is on lm-tasks, which are closer to COT than MC benchmarks.

Was able to run gsm with:

helm-run --run-entries gsm:model=openai/gpt2 \
 --suite test \
 --output-path src/data/helm/ \
 --max-eval-instances 1

But the metrics return 0. There is an error somewhere i can't find... Too tired, tomorrow i'm testing this with better models and seeing if it this fixes by itself.

helm-run --run-entries gsm:model=microsoft/phi-3.5-mini-instruct \
 --suite test \
 --output-path src/data/helm/ \
 --max-eval-instances 1

## August 20th 2025:

Succesfully ran:

helm-run --run-entries gsm:model=microsoft/phi-3.5-mini-instruct \
 --suite test \
 --output-path src/data/helm/ \
 --max-eval-instances 1

With the three entropy based metrics. All that's left is running it on all the relevant datasets and computing the results.

TODO: Run the evaluations and present them in a cute way.

## August 21th 2025:

I'm running helm lite with 100 instances per dataset on the phi 3.5 model.

helm-run --conf-paths src/eval-engine/src/helm/benchmark/presentation/run_entries_lite.conf \
 --models-to-run microsoft/phi-3.5-mini-instruct \
 --suite helm-lite-naive \
 --output-path src/data/helm/ \
 --max-eval-instances 100

And also introduced a "max nll" metric that is just the maximum NLL across the generated sequence.

I'm gonna need to use an external VERY SMALL llm agent to parse the results, I can just use whatever is reported as a placeholder in the meanwhile.

Successfully ran everything except for math, raft, quac and natural_qa.

To test math:

helm-run --run-entries math:model=microsoft/phi-3.5-mini-instruct,subject=number_theory,level=1,use_chain_of_thought=True \
 --suite test \
 --output-path src/data/helm/ \
 --max-eval-instances 10

To test natural_qa:

helm-run --run-entries natural_qa:model=microsoft/phi-3.5-mini-instruct,mode=openbook_longans,output_format_instructions=natural_qa \
 --suite test \
 --output-path src/data/helm/ \
 --max-eval-instances 10

Cool, fixed! Now running everything on a differnent config file for instruct models:

helm-run --conf-paths src/eval-engine/src/helm/benchmark/presentation/run_entries_lite_20240424_instruct.conf \
 --models-to-run microsoft/phi-3.5-mini-instruct \
 --suite helm-lite-instruct \
 --output-path src/data/helm/ \
 --max-eval-instances 100

Tomorrow I want to have an actual table of results. The work is most likely going to just be presenting the results from today and possibly re-running on some scenarios that were not able to run.

After that: Either implement ASEU or the domain clustering algorithm. Gonna talk it with my director.

Had errors on:

natural_qa:mode=openbook_longans,model=microsoft_phi-3.5-mini-instruct
med_qa:model=microsoft_phi-3.5-mini-instruct

Trying to run them with:

helm-run --run-entries natural_qa:mode=openbook_longans,model=microsoft/phi-3.5-mini-instruct \
 --suite helm-lite-instruct \
 --output-path src/data/helm/ \
 --num-threads 1 \
 --max-eval-instances 100

Which was unsuccesful...

helm-run --run-entries med_qa:model=microsoft/phi-3.5-mini-instruct \
 --suite helm-lite-instruct \
 --output-path src/data/helm/ \
 --max-eval-instances 100

Which worked! I have enough to process the results.

math:subject=precalculus,level=1,use_official_examples=False,use_chain_of_thought=True,model=microsoft_phi-3.5-mini-instruct

helm-run --run-entries math:subject=precalculus,level=1,use_official_examples=False,use_chain_of_thought=True,model=microsoft/phi-3.5-mini-instruct \
 --suite helm-lite-instruct \
 --output-path src/data/helm/ \
 --max-eval-instances 10

I had some errors on the math dataset (so problematic). It appends some tool use at the end of the generated sequence. This heavily distorts the metrics, but whatever, i may just ignore that dataset and focus on the ones that were able to run.

Everything but memory issues was solved! I'm re-running the evaluations on the phi 3.5 mini instruct model with this command:

helm-run --conf-paths src/eval-engine/src/helm/benchmark/presentation/run_entries_lite_20240424_instruct.conf \
 --models-to-run microsoft/phi-3.5-mini-instruct \
 --suite helm-lite-instruct \
 --output-path src/data/helm/ \
 --max-eval-instances 100

I increased the max tokens by 300 in all instances to ensure responses are not cut off. Nothing else to do but wait!

Everything ran except for:

"natural_qa:mode=openbook_longans,model=microsoft_phi-3.5-mini-instruct"

helm-run --run-entries natural_qa:mode=openbook_longans,model=microsoft/phi-3.5-mini-instruct \
 --suite helm-lite-instruct \
 --output-path src/data/helm/ \
 --num-threads 1 \
 --max-eval-instances 100

## August 22th 2025:

I was able to get a table with the metrics, but without either AUROC or some benchmark/accuracy metric. 

TODO:

- Write a script using gemini flash that produces an actual benchmark for each task.
- Compute the AUROC for each metric, grouped by scenario, domain and globally.
- Make all benchmarks COT where possible so that they can be compared.

# August 23th 2025:

Wrote the thesis plan and found where to make the benchmarks COT. I'm also removing the translation task since its. I may also add a bunch of other interesting datasets.

TODO:

1. Make all benchmarks COT where possible
2. Compute the benchmarks and AUROC for each metric
3. Present the results for the naive metrics

After that I can start thinking about ASEU and domain clustering, possibly even using other "expensive" metrics just to fill the table with different options! Looks promising.
The COT can be added in the run_expander.py file.

## August 25th 2025:

So, to make the benchmarks comparable, i'm using the COT 0-shot version. That can be done by setting max_train_instances to zero.

I successfully made the evaluations COT with zero shot! All that's left is running the benchmarks and having an llm evaluate the results.

Made a custom config!

helm-run --conf-paths src/eval-engine/src/helm/benchmark/presentation/run_entries_custom.conf \
 --models-to-run microsoft/phi-3.5-mini-instruct \
 --suite helm-lite-custom \
 --output-path src/data/helm/ \
 --max-eval-instances 150

Also wrote the necessary code to compute the benchmarks using gemini 2.5 flash. All that is left is running it and computing the AUROC for each metric. Everything is already set up to play with other metrics and instruct models.

I also debated with myself about the merits of creating an eval-engine from scratch. It may be a good idea to do it eventually, but right now its too much work when i can work with helm... But on the other hand it will get messy when I include black box metrics. 

Idea:

- 1. By default: COT 0-shot with open generation
- 2. Custom prompt by domain
- 3. Enable multiple evaluation strategies (necessary for black box)
- 4. Implement all relevant metrics
    - shannon, predictive, nll, max nll
    - internal cosine sim (internal & external), semantic entropy, bertscore

Will take me 2 weeks probably but it can be really worth it! Better talk it with my director after showcasing the results first.

I can also add a bunch more datasets by making it custom! Possibly most of the phi 3.5 ones on top of helm.

I have debugged everything. At night i'm running the evaluations and tomorrow i'm analysing the results. Not doing anything on wensday, gotta study.

## August 26th 2025:

Finally some results, but they are not so promising. The entropy for different tasks is very different. Non-math benchmarks have intrinsically more logprob entropy, whereas math is lower. This makes comparing performance between benchmarks very hard. Even within benchmarks, the auroc is very close to 0.5 other than in some specific cases (gsm or narrative_qa). I am also not seeing any type of correlation between the metrics themselves and the bechmark score. 

This is a result in itself though, white box metrics are NOT suitable for finding domains in which the performance is low. They are primarily determined by the type of task. Of course more data is needed to affirm this in a paper, but i am quite positive of these results. I can also obtain them in paralell to the next step.

The results kinda also force the next step, which is trying black box metrics on many more datasets. I am 100% building a simple framework and ditching crfm-helm.

Ideally I get promising results with black box / ensemble metrics, which in turn motivates implementing ASEU and domain clustering...

## August 27th 2025:    

Possible title: 

Clarity-Oriented

“Estimating Answer Trustworthiness from LLM Internal States”

“Real-Time Confidence Scoring for LLM Responses Without Supervision”

“On-the-Fly Reliability Estimation in LLMs via Internal Signals”

Novelty-Emphasizing

“Trust Signals: Leveraging Internal Dynamics of LLMs for Confidence Estimation”

“Self-Reflective Models: Predicting Answer Reliability from Hidden States”

“Hallucination Detection from Within: Confidence Estimation via LLM Internals”

Scientific/Compact

“Unsupervised Confidence Estimation in LLMs”

“Intrinsic Confidence: A Self-Supervised Approach to Answer Trustworthiness”

“Hidden-State Metrics for Real-Time LLM Confidence”

Talked with my director, he liked the results and the conclusion that the logprob metrics are not enough. He suggested trying different metrics based on the variation between the distribution along the transformer layers. There are quite a lot of metrics that can be derived from that. 

The priority is playing with this metric as a candidate in the context of gsm and math benchmarks. After that i may go with ASEU or the black box metrics.

Possible metrics:

- Average early stop layer accoring to some criterion, such as hidden state or softmax + threshold or "p-value" for exiting a given layer

- KL-divergence metrics (maybe for some subset such as last 1/2 or 1/4 layers):
    - U₁ = (1/L) * Σᵢ₌₁ᴸ KL(pᵢ || pₗ)
    - U₂ = Var(KL(p₁ || pₗ), KL(p₂ || pₗ), ..., KL(pₗ₋₁ || pₗ))
    - U₄ = KL(p_early || p_late) (p_early and late averages of some quarters)
    - U₅ = Var(H(p₁), H(p₂), ..., H(pₗ))

## August 29th 2025

Thought something soooo cool. All these token level uncertainty estimations get dilluted when averaging over long sequences. That may be one of the reasons why the AUROC was reasonably high for gsm8k and narrative-qa, but not for other metrics. There are many ways of pooling and averaging to ignore low-entropy tokens:

- Entropy / max prob weighting
- Top k proportion pooling (maybe something dynamic too?)

This applies to ALL uncertainty metrics! This opens a pandora box of experimentation! I can take the pooling as a hyperparameter and try a bunch of metrics such as the ones above.

The idea is as simple as defining a token-level metric, and then normalizing/pooling.

## September 1st 2025

Debugged the layer evolution metrics and the last layer logprobs metrics. Also made them batch friendly. Tomorrow i begin writing the early exit metrics.

## September 2nd 2025

Developed the early exit metric (did not debug it though). I'm leaving the t-value for later, since i already have quite a lot to experiment with.

Next step is running gsm and computing the metrics. If they have a highish AUROC, then its worthwhile to compare them with black box metrics, and with other benchmarks...

## September 4th 2025

Wrote the run script. Next week (have a final on monday, so not touching the thesis) i'm debugging the code, making sure everything runs properly and running the evaluations on gsm8k. I'd love to have some results by the 11th!

## September 9th 2025

To run the tests this is the command:

python -m src.engine.run \
  --dataset_name "gsm8k" \
  --model_name "microsoft/Phi-3-mini-4k-instruct" \
  --suite "test" \
  --result_path "./src/runs" \
  --temperature 0.5 \
  --max_length 512 \
  --device "cpu" \
  --limit 1

Today I debugged the early exit metric and the run function.

I noticed that the weighting is not really a "weighted average" in the proper sense (weights dont add up to 1). That may skew the results quite a bit. I modified it so that it performs a proper weighted average with l1 normalization. 

All that is left is running the results and seeing whatever it returns. I absolutely want to implement attention weighting, which is quite more sophisticated (32 layers, each with 32 attention heads). Attentions from each layer can themselves be weighted, possibly by the amount of state or softmax variation!

TODO:

- Run on gsm and presents the results with what I already have.
- Implement attention weighting and re-run.
- If results are good, compare with black box metrics.

## September 10th 2025

I ran:

python -m src.engine.run \
  --dataset_name "gsm8k" \
  --model_name "microsoft/Phi-3-mini-4k-instruct" \
  --suite "validation" \
  --result_path "./src/data/runs" \
  --temperature 0.5 \
  --max_length 1024 \
  --device "cuda:0" \
  --limit 300

And stored the results. Tomorrow i'm plotting the auroc in a cute table, and talking with my director about how to approach the attention design. If the attention and other metrics provide a good auroc, all that is left (as far as not-writing goes) is re-running this on many more benchmarks and analysing the results!

## September 11th 2025  

https://arxiv.org/pdf/1804.07781
https://aclanthology.org/2020.acl-main.385.pdf
https://arxiv.org/abs/1902.10186

Done today: 

- Store tensors for 100 items of gsm8k
- Evaluate on those 100 items with gemini flash

Ran: 

python -m src.engine.run \
  --dataset_name "gsm8k" \
  --model_name "microsoft/Phi-3-mini-4k-instruct" \
  --suite "validation" \
  --result_path "./src/data/runs" \
  --temperature 0.5 \
  --max_length 1024 \
  --store_tensors True \
  --store_metrics False \
  --device "cuda:0" \
  --limit 120

which now simply stores the tensors, rather than compute the metrics. The 100 samples take about 26gb of space.

TODO for tomorrow:

1. Write the grid_run.py script to run all the metrics from the stored tensors.
2. Compute AUROC for the entire grid.
3. Understand the attention propagation paper thing xdxd.

## September 12th 2025

Wrote the grid_run.py script:

python -m src.metrics.grid_run \
    --model_name "microsoft/Phi-3-mini-4k-instruct" \
    --datafile "./src/data/runs/validation/gsm8k_microsoft_Phi-3-mini-4k-instruct_20250911-143125.pt" \
    --output_file "./src/data/results/gsm8k_phi3_mini_metrics_layer_evolution.json"  \
    --metrics "layer_evolution_mean_kl_divergence" "layer_evolution_var_kl_divergence" "layer_evolution_mean_shannon_entropy" "layer_evolution_var_shannon_entropy"

That computes the auroc for the specified metrics (100 items). Also ran it.

The conclusion is that logit based metrics completly outperform early exit and layer evolution metrics. The pooling mechanism is also quite influential. For logit based-metrics without weighting, the auroc considerably decreases the more of the sequence is taken into account, while the top values only average the top 5% highest entropy tokens. Weights are also significant, with prob/entropy weighing slightly increasing the auroc (but not enough data is present to properly conclude this)

Layer evolution variance can be quite good for shannon variance for the last n layers!

For other metrics, the auroc is highest on the raw average without weighting. 

I believe this simplifies the problem quite a bit. We already have the token-level metric (shannon entropy). The issue is now finding the way to ensemble this. If attention flow works properly, all thats left is comparing with black box metrics, telling the story, and writing the thesis.

## September 15th

Found this paper: https://arxiv.org/pdf/2409.19001v1, which applies attention rollout ideas to decode-only models! May be super useful. 

Today I implemented the attention rollout weighting. I normalized the attention weights by the amount of tokens that can attend to each token. This consists of basically dividing the i-th column by (n - i), where n is the length of the sequence, and then normalizing so that it adds up to 1.

The normalization step is impossibly ugly, but otherwise the entire attention resides in the first token.

The following todos are computing the auroc with these new weights, and implementing other attention weighting schemes (such as the ones from the paper).

## September 16th 

Included norm weighting from the paper, and also added a cosine difference weighting. For the normal rollout, i enabled an option that lets me select the proportion of weight to attention, and the proportion to the identity matrix (residual connection).

There are sooo many more things to try regarding the attention metrics. But I need to re-run everything to get the new internal signals. I have no connection to my director's pc and no battery, so i'm going to wait until tomorrow to re-run everything.

python -m src.engine.run \
  --dataset_name "gsm8k" \
  --model_name "microsoft/Phi-3.5-mini-instruct" \
  --suite "validation" \
  --result_path "./src/data/runs" \
  --temperature 0.5 \
  --max_length 1024 \
  --store_tensors True \
  --store_metrics False \
  --device "cuda:0" \
  --limit 110

## September 17th

Running:

python -m src.metrics.grid_run \
    --model_name "microsoft/Phi-3-mini-4k-instruct" \
    --datafile "./src/data/runs/validation/gsm8k_microsoft_Phi-3.5-mini-instruct_20250916-210355.pt" \
    --output_file "./src/data/results/gsm8k_phi3.5_mini_shannon_attention_weight_end_pool_weight.json"  \
    --metrics "logits_shannon_entropy"

Results are bad, the attention weighting lowers the auroc in all configurations. It makes sense since rollout is sooo noisy even with improvmenents.

I did find a sota technique that relies on backprop. It is definitely more expensive, but it may be worth it to try it out and see if it works:

https://lxt.readthedocs.io/en/latest/quickstart.html

Regardless of this, the attention rollout improvements are interesting in themselves. While they can't be applied to autoregressive models, they CAN be applied to bert like models and may just improve results (possible paper?).

## September 18th

Taking a break for today. This is a recap of what i talked with my director:

- Estudio de ablacion donde divido los ejemplos por el largo total.

Leads:
    - Comparacion curvas
    - LPX para tener una buena estimacion de la relevancia de los tokens
    - Hacer attention flow sobre el encoder

Empiezo comparando las curvas, y veo a partir de ahi.

Plotie las curvas de shannon y encontre que es falsa la idea de que los primeros tokens son los mas inciertos. Las curvas son hiper picudas y no parecen tener mucho patron a nivel donde estan los picos. Calculando metricas en funcoin de los picos, encuentro que el acg height es muy similar en los casos positivos y negativos. Lo que varia es la cantidad de picos y su densidad.

Si puedo pesar esos picos por la atencion que se les da, puedo tener una re estimacion... 

LRP puede ser promisorio, por hacer algo mas que nada...

TODO: Ver (graficamente) con heatmaps los metodos interpretabilidad atencion. 

## September 22th

I visualized the attention influence methods on a heatmap. It looks super cool. Tomorrow i'm playing with using them to weight the entropy.

I also introduced some receptive field normalization which is promising!

Next step is weighing the entropies with these and analyze how the distributions compare...


## September 23th

python -m src.metrics.grid_run \
    --model_name "microsoft/Phi-3-mini-4k-instruct" \
    --datafile "./src/data/runs/validation/gsm8k_microsoft_Phi-3.5-mini-instruct_20250916-210355.pt" \
    --output_file "./src/data/results/gsm8k_phi3.5_mini_shannon_attention_quantile75.json"  \
    --metrics "logits_shannon_entropy" \
    --weighting_methods "raw_mean" \
    --aggregation_methods "all_sequence"

Okay, so the attention weighting is not working. It improves with receptive field normalization but it still does not beat the raw mean. I also tested a bunch of metrics based on the cohens denominator, and exclusive logit based metrics always win.

So, i can make a really good argument for the entropy distribution being an amazing feature for performance and halucination prediction. But can i really say more? I'm starting to think that its going to be quite hard incrementing the auroc significantly...

I for sure need more data... Let's see what my director says.

## September 25th

1. Compare cross-dataset

Para contar la historia:
  -1. A study metrics for task agnostic llm-performance:
    Probe banda de cosas y me da esto

  -2. Token-level Shannon entropy distribution is a strong predictor of llm-performance:
    Me quedo con la que anda mejor y hago un estudio de ablacion 

  -3. Efficent metrics for task agnostic llm-performance estimation:
    Me quedo con las que andan bien y son baratas

2. Train the metamodel on shanon entropy distribution features

The TODOs are as following:

- Compute the shannon entropies for MATH and GSM8K, analyze how they compare.
- Train a logistic regression model on the shannon entropy distribution features. The regression classifies if the answer is correct or not.

# October 8th

Took a looong break to work in AITW and prepare the language paradigms exam.

The TODO for the next month is:

- Decide on a rollout algorithm to store influences
- Re-run the evaluations on gsm8k and math with the rollout algorithm
- Visualize the results
- Train a logistic regression model on the shannon entropy distribution features for a haluccination detection task
- See the results on 5 different datasets, and compare them with the benchmarks

Found out that the attention rollout weighting is actually amazing. I tried doing RECEPTIVE FIELD normalization, and ALSO averaging the final [seq, seq] tensor also by its respective receptive field on the first dimension, and that reaches an AUROC of 0.8-0.814 depending on the algorithm used. Surprisingly, the best results comes from the regular rollout algorithm with.

I tried other receptive field normalizations but all of them make the results worse. 

Next step is re-running everything on math and gsm8k (TRAIN) and storing:
    - Shanon entropies
    - Attention weights (with and without receptive field normalization):
        - Raw rollout 0.5, 0.7, 0.9 residual
        - Norm rollout
        - Projection rollout

After that I can visualize the results and see how they compare on both datasets.

# October 9th

Ran:

python -m src.engine.store \
  --dataset_name "gsm8k" \
  --model_name "microsoft/Phi-3.5-mini-instruct" \
  --suite "gsm-complete-test" \
  --result_path "./src/data/runs" \
  --temperature 0.5 \
  --max_length 1024 \
  --device "cuda:0" \
  --store_attention_influence True \
  --store_logprobs True

and

python -m src.engine.store \
  --dataset_name "math" \
  --model_name "microsoft/Phi-3.5-mini-instruct" \
  --suite "math-complete-test" \
  --result_path "./src/data/runs" \
  --temperature 0.5 \
  --max_length 1024 \
  --device "cuda:0" \
  --store_attention_influence True \
  --store_logprobs True

It store the shannon entropies and different attention influence weights.

# October 10th

Ran:

python -m src.engine.evaluate --suite "gsm-test"

# October 15th

5 days ago I ran everything on gsm-test and got an auroc of 0.75, which is better than just averaging but only marginally. I went on a tangent of trying different hyperparameters for the rollout algorithm. I ended up with these possibilities:

Algorithms:
    - Raw rollout (different proportions)
    - Norm and projection rollout
    - Macs algorithm (different epsilons)
    - Raw mean, and raw mean with relevance (tomorrow)

Normalizations:
    - Receptive field normalization (2 of them)
    - No normalization

Aggregations:
    - Raw mean
    - Receptive field norm mean
    - Max
    - + different distribution statistics

So... I should write the pipeline for trying that with different models and datasets.

After that its just writing and obtaining more data for each ensemble.
