# Advanced Image Captioning Transformer with Reinforcement Learning Optimization in PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danplotkin/image_captioning_with_scst/blob/main/ImageCaptioner.ipynb)

## About

This repository contains a Colab Notebook that implements a two-step training process to train an image captioner.

We implement a baseline transformer image captioning model using the CPTR architecture, as described in the research paper "[CPTR: Full Transformer Network for Image Captioning](https://arxiv.org/pdf/2101.10804)" by Liu et al. (2021).

<img src='https://media.licdn.com/dms/image/C4D12AQGA3qFX3peTbw/article-cover_image-shrink_720_1280/0/1648387317335?e=2147483647&v=beta&t=4VOpEV8ptM4B4Q0UTZJUWqv4QFQvIuCubBoQLzJazds' width='800'>

We optimize this baseline model with Self-critical Sequence Training (SCST), proposed in "[Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563)" by Rennie et al. (2016), which is an improved variant of the popular REINFORCE algorithm. Below is a depiction of this process from the original paper:

<img src='https://github.com/danplotkin/image_captioning_with_scst/blob/main/images/SCST.png'>

Instead of using CIDEr as our reward function, we use the METEOR score, introduced in "[Meteor: An Automatic Metric for MT Evaluation with High Levels of Correlation with Human Judgments](https://www.cs.cmu.edu/~alavie/METEOR/pdf/Banerjee-Lavie-2005-METEOR.pdf)".

## Dataset

We use the [Flickr8K dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k). Our random train-val-test splits are shown below:

- **Number of training examples:** 23890
- **Number of validation examples:** 5975
- **Number of test examples:** 7470

## Baseline Training

### Configurations

#### Hyperparameters

- **Batch Size:** 40
- **Embedding Dimensions:** 768
- **Number of Decoder Layers:** 4
- **Number of Attention Heads:** 12
- **Dense Neurons:** 1536
- **Max Epochs:** 15

#### Learning Rate Schedule and Early Stopping

- We use a linear warmup learning rate method that warms up to the rate of 1e-4, which then decays using cosine decaying.
- Our early stopping procedure has a patience of 1 and reverts to the best weights based on the validation loss.

#### Pretrained Components

- **Tokenizer:** `distilbert-base-uncased`
- **ViT:** `google/vit-base-patch16-384`

#### Hardware

- **GPU:** L4 Colab GPU

#### Loss Function and Metric

- **Loss Function:** Zero-masked Categorical Cross Entropy Loss (XE)
- **Metric:** Zero-masked Accuracy

### Results

#### Early Stopping

Our training ended at epoch 10, and we reverted back to weights used at the end of epoch 9.

#### Train and Validation Loss by Epoch

<img src='https://github.com/danplotkin/image_captioning_with_scst/blob/main/images/CPTR_LOSS.png'>

#### Train Validation Accuracy by Epoch

<img src='https://github.com/danplotkin/image_captioning_with_scst/blob/main/images/CPTR_ACCURACY.png'>

## Self-critical Sequence Training (SCST)

### Configurations

#### Hyperparameters

- **Epochs:** 4
- **Batch Size:** 12

#### Learning Rate Schedule

- **Initial Learning Rate:** 1e-5. We then decay the learning rate by 0.5 for the last two epochs.

#### Loss Function and Rewards

We use METEOR score as our non-differentiable reward function. We aim to maximize reward by minimizing the loss function with the following gradient computation:

$$
\nabla_{\theta} L(\theta) = - \mathbb{E} _ {w^s \sim p{\theta}} \left[ (r(w^s) - b) \nabla_{\theta} \log p_{\theta}(w^s) \right]
$$

where:  
- $\theta$ represents the model parameters,
- $w^s$ is a sampled sequence from the model's probability distribution $p_{\theta}$,
- $r(w^s)$ is the reward associated with the sequence $w^s$,
- $b$ is the baseline reward, which is typically the reward of a baseline sequence (e.g., the sequence generated by the current model without sampling).

#### Hardware

- **GPU:** Colab L4 GPU

### Results

#### Batched-test Single-reference METEOR Before and After SCST

<img src='https://github.com/danplotkin/image_captioning_with_scst/blob/main/images/SCST_TEST_RESULT.png'>

## Final Scores

We use **beam search** to decode our captions for final evaluation and generation, with a beam size of 3. A simple normalized function is used as the score function for beam search. It is defined as:

$$
\text{score}(y) = \log P(y \mid x) = \frac{1}{T} \sum_{i=1}^{T} \log P(y_i \mid y_1, \ldots, y_{i-1}, x)
$$

Where at each step, we selected the 3 largest values of $\text{score}(y)$.

| METEOR | BLEU 1 | BLEU 2 | BLEU 3 | BLEU 4 |
|--------|--------|--------|--------|--------|
| 0.4618 | 0.5419 | 0.3892 | 0.2627 | 0.1698 |

## Generated Captions on Unseen Images

Below are 5 randomly sampled captions from our test dataset:

<img src='https://github.com/danplotkin/image_captioning_with_scst/blob/main/captions/scst_cap_1.png' width='500'>
<img src='https://github.com/danplotkin/image_captioning_with_scst/blob/main/captions/scst_cap_2.png' width='500'>
<img src='https://github.com/danplotkin/image_captioning_with_scst/blob/main/captions/scst_cap_3.png' width='500'>
<img src='https://github.com/danplotkin/image_captioning_with_scst/blob/main/captions/scst_cap_4.png' width='500'>
<img src='https://github.com/danplotkin/image_captioning_with_scst/blob/main/captions/scst_cap_5.png' width='500'>

It can be seen that the captions are not perfect, but for the most part are able to understand the most of the captions to a good degree.

### Further Improvements

We can further improve the performance of the model by:

- Using a larger dataset like Flickr32k or MSCOCO to train our model.
- Making our model more complex with more decoder layers.
- Running our SCST for more epochs.

### Citations

```
@article{DBLP:journals/corr/RennieMMRG16,
  author       = {Steven J. Rennie and
                  Etienne Marcheret and
                  Youssef Mroueh and
                  Jerret Ross and
                  Vaibhava Goel},
  title        = {Self-critical Sequence Training for Image Captioning},
  journal      = {CoRR},
  volume       = {abs/1612.00563},
  year         = {2016},
  url          = {http://arxiv.org/abs/1612.00563},
  eprinttype    = {arXiv},
  eprint       = {1612.00563},
  timestamp    = {Tue, 23 Jul 2019 16:55:13 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/RennieMMRG16.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@article{DBLP:journals/corr/abs-2101-10804,
  author       = {Wei Liu and
                  Sihan Chen and
                  Longteng Guo and
                  Xinxin Zhu and
                  Jing Liu},
  title        = {{CPTR:} Full Transformer Network for Image Captioning},
  journal      = {CoRR},
  volume       = {abs/2101.10804},
  year         = {2021},
  url          = {https://arxiv.org/abs/2101.10804},
  eprinttype    = {arXiv},
  eprint       = {2101.10804},
  timestamp    = {Wed, 12 Oct 2022 13:48:47 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2101-10804.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@article{DBLP:journals/corr/VaswaniSPUJGKP17,
  author       = {Ashish Vaswani and
                  Noam Shazeer and
                  Niki Parmar and
                  Jakob Uszkoreit and
                  Llion Jones and
                  Aidan N. Gomez and
                  Lukasz Kaiser and
                  Illia Polosukhin},
  title        = {Attention Is All You Need},
  journal      = {CoRR},
  volume       = {abs/1706.03762},
  year         = {2017},
  url          = {http://arxiv.org/abs/1706.03762},
  eprinttype    = {arXiv},
  eprint       = {1706.03762},
  timestamp    = {Sat, 23 Jan 2021 01:20:40 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/VaswaniSPUJGKP17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@article{article,
  author       = {Lavie, Alon and Agarwal, Abhaya},
  year         = {2007},
  month        = {07},
  pages        = {228-231},
  title        = {METEOR: An automatic metric for MT evaluation with high levels of correlation with human judgments}
}
```

