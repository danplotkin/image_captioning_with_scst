# Advanced Image Captioning Transformer with Reinforcement Learning Optimization in PyTorch

## About

This repository contains a Colab Notebook that implements a two-step training process to train an image captioner.

1. We implement a baseline transformer image captioning model using the CPTR architecture, as described in the research paper "[CPTR: Full Transformer Network for Image Captioning](https://arxiv.org/pdf/2101.10804)" by Liu et al. (2021).
<img src='https://media.licdn.com/dms/image/C4D12AQGA3qFX3peTbw/article-cover_image-shrink_720_1280/0/1648387317335?e=2147483647&v=beta&t=4VOpEV8ptM4B4Q0UTZJUWqv4QFQvIuCubBoQLzJazds' width='800'>

2. We optimize this baseline model with Self-critical Sequence Training (SCST), proposed in "[Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563)" by Rennie et al. (2016) which is an improved varient of the popular REINFORCE algorithm. Below is a depiction of this process from the original paper:
<img src='https://github.com/danplotkin/image_captioning_with_scst/blob/main/images/SCST.png'>
Instead of using CIDER as our reward metric, we use the METEOR score, introduced in "[Meteor: An Automatic Metric for MT Evaluation with HighLevels of Correlation with Human Judgments](https://www.cs.cmu.edu/~alavie/METEOR/pdf/Banerjee-Lavie-2005-METEOR.pdf)". 



