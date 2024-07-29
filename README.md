# Advanced Image Captioning Transformer with Reinforcement Learning Optimization in PyTorch

## About

This repository contains a Colab Notebook that implements a two-step training process to train an image captioner.

We implement a baseline transformer image captioning model using the CPTR architecture, as described in the research paper "[CPTR: Full Transformer Network for Image Captioning](https://arxiv.org/pdf/2101.10804)" by Liu et al. (2021).

<img src='https://media.licdn.com/dms/image/C4D12AQGA3qFX3peTbw/article-cover_image-shrink_720_1280/0/1648387317335?e=2147483647&v=beta&t=4VOpEV8ptM4B4Q0UTZJUWqv4QFQvIuCubBoQLzJazds' width='800'>

We optimize this baseline model with Self-critical Sequence Training (SCST), proposed in "[Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563)" by Rennie et al. (2016) which is an improved varient of the popular REINFORCE algorithm. Below is a depiction of this process from the original paper:
   
<img src='https://github.com/danplotkin/image_captioning_with_scst/blob/main/images/SCST.png'>

Instead of using CIDER as our reward metric, we use the METEOR score, introduced in "[Meteor: An Automatic Metric for MT Evaluation with HighLevels of Correlation with Human Judgments](https://www.cs.cmu.edu/~alavie/METEOR/pdf/Banerjee-Lavie-2005-METEOR.pdf)". 

## Baseline Training

### Configurations

#### Hyperparameters
* Batch Size: 40
* Embedding Dimentions: 768
* Number of Decoder Layers: 4
* Number of Attention Heads: 12
* Dense Neurons: 1536
* Max Epochs: 15

#### Learning Rate Schedule and Early Stopping
* We use a linear warmup learning rate method that warms up to the rate of 1e-4, which then decays using cosign decaying.
* Our early stopping procedure has a patience of 1, and reverts to the best weights based on the validation loss.

#### Pretrained Components
* Tokenizer: `distilbert-base-uncased`
* ViT: `google/vit-base-patch16-384`

#### Hardware
* GPU: L4 Colab GPU

#### Loss Function and Metric
* Loss Function: Zero-masked Categorical Cross Entropy Loss (XE)
* Metric: Zero-mased Accuracy

### Results

#### Early Stopping
Our training ended at epoch 10, and we reverted back to weights used at the end of epoch 9.

#### Validation Loss by Epoch

<img src='https://github.com/danplotkin/image_captioning_with_scst/blob/main/images/CPTR_LOSS.png'>

#### Validation Accuracy by Epoch

<img src='https://github.com/danplotkin/image_captioning_with_scst/blob/main/images/CPTR_ACC.png'>

## Self-critical Sequence Training (SCST)

### Configurations

#### Hyperparameters
* Epochs: 2
* Batch Size: 12

#### Learning Rate Schedule
* Initial Learning Rate: 5e-6
* Decay LR by 0.5 after first epoch.

#### Loss Function and Rewards
* We use METEOR score as our non-differentiable reward function. We aim to maximize reward by minimizing the loss function with the following gradient computation:
  
$$\nabla_{\theta} L(\theta) = - \mathbb{E}{w^s \sim p{\theta}} \left[ (r(w^s) - b) \nabla_{\theta} \log p_{\theta}(w^s) \right]$$

#### Hardware
* GPU: Colab L4 GPU
