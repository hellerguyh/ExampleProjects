# Implementation of "Decomposable Attention Model for Natural Language Inference" - Parikh et al.

This code is an implementation for the Decomposable Attention Model for NLP. The code implemented using Pytorch, and can run either over Google-Colab or locally

## How to run on Google-Colab

1. Create a folder in drive and extract the SNLI dataset in it
2. Point to the folder using FOLDER_PATH
3. Make sure USE_CUDA = True, RUNNING_LOCAL = False
   1. for running locally without GPU choose RUNNING_LOCAL = True
4. Run colab_preq.py 
5. Run nn.py 

## Results

85.83% Accuracy

## Implementation Notes

**Embedding**
We've used GloVe-840 which has 840B tokens. We've also tried using GloVe6B but it reached significantly lower results. Because of memory requirements, we couldn't run the model with the vocabulary as is. To solve this problem we replaced the GloVe-840 vocabulary with a vocabulary that only includes words from test, train and validation datasets. 

To tokenize the words we used SpaCy tokenizer. SpaCy showed better results than just separating the words according to spaces.

**Optimizers and Loss**

We've used Cross-Entropy loss and Adagrad optimizer with weight decay of 0.000002, initial accumulator value of 0.1, and learning rate of 0.05. Setting the initial accumulator value decreased the convergence time. 

Initial results which used Adam optimizers showed lesser performance.

**Batch size** - 

We've used 32 as batch size. Batch sizes 4 and 50 showed lesser results

## Paper
The paper suggests a new method to perform Natural Language Inference. In this task, the model needs to determine the relationship between two sentences (entailment/natural/contradiction). The paper claims that to resolve this relationship it's enough to align phrases with similar (or contradicting) semantic, without the need to fully understand the whole sentence. A model that doesn't need to understand the whole meaning of the sentence can be lighter than a model that does.

The model could be described by the following equations. Each sentence is attended according to the other sentence. In this way phrases with similar (or contradicting) meaning will be in matching places in vectors ($\hat{a},\beta$)  and ($\hat{b},\alpha$). aligned sentences are compared using a feed forward network. Intuitively each element in $v^a, v^b$ will encode whether the matching phrases represent entailment/natural/contradiction. This score can be accumulated and then used to classify the relationship.
$$
\begin{align}
{}& \text{Inputs:}
\\{}& a = (a_1,...,a_{l_a}), a_i\in\R^d \text{ - sentence}
\\{}& b = (b_1,...,b_{l_a}), b_i \in \R^d \text{ - sentence}
\\{}& y = (y_0,y_1,y_2), y_i \in \{0,1\}
\\
\\{}& \text{Attention}
\\{}& \hat{a} = a, \hat{b} = b
\\{}& e_{ij} = FFN(a_i)^TFFN(b_j)
\\{}& w^\alpha_{ij} = \frac{e_{ij}}{\sum_{k=1}^{l_a}e_{kj}}
\\{}& w^\beta_{ij} = \frac{e_{ij}}{\sum_{k=1}^{l_a}e_{ki}}
\\{}& \alpha_{j} = \sum_{k=1}^{l_a}w^{\alpha}_{kj}\hat{a}_k
\\{}& \beta_{i} = \sum_{k=1}^{l_b}w^{\beta}_{ik}\hat{b}_k
\\{}& v^{a}_i = FFN(\hat{a}_i,\beta_i), \ \ 1 \leq i \leq l_a
\\{}& v^{b}_j = FFN(\hat{b}_j,\alpha_j), \ \ 1 \leq j \leq l_a
\\
\\{}& \text{Compare & Aggregate}
\\{}& v_a = \sum_{i=1}^{l_a}v_i^a
\\{}& v_b = \sum_{j=1}^{l_b}v_j^b
\\{}& \text{Predict}
\\{}& \hat{y} = FFN([v_a,v_b])

\end{align}
$$



