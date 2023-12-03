# 多模态学习调研

By[Xiangyang HU](232050209@hdu.edu.cn)

## 目录

* [综述](#survey-papers)
* [核心领域](#core-areas)
  * [多模态表示（Multimodal Representations）](#multimodal-representations)
  * [多模态融合（Multimodal Fusion）](#multimodal-fusion)
  * [多模态对齐（Multimodal Alignment）](#multimodal-alignment)
  * [多模态预训练（Multimodal Pretraining）](#multimodal-pretraining)
  * [多模态翻译（Multimodal Translation）](#multimodal-translation)
  * [跨模态检索（Crossmodal Retrieval）](#crossmodal-retrieval)
  * [多模态共同学习（Multimodal Co-learning）](#multimodal-colearning)
  * [模态缺失（Missing or Imperfect Modalities）](#missing-or-imperfect-modalities)
  * [多模态模型分析（Analysis of Multimodal Models）](#analysis-of-multimodal-models)
  * [知识图谱（Knowledge Graphs and Knowledge Bases）](#knowledge-graphs-and-knowledge-bases)
  * [可解释的学习（Intepretable Learning）](#intepretable-learning)
  * [生成学习（Generative Learning）](#generative-learning)
  * [半监督学习（Semi-supervised Learning）](#semi-supervised-learning)
  * [自监督学习（Self-supervised Learning）](#self-supervised-learning)
  * [语言模型（Language Models）](#language-models)
  * [对抗学习（Adversarial Attacks）](#adversarial-attacks)
  * [少样本学习（Few-Shot Learning）](#few-shot-learning)
  * [公平公正问题（Bias and Fairness）](#bias-and-fairness)
  * [人在回路学习（Human in the Loop Learning）](#human-in-the-loop-learning)
* [架构](#architectures)
  * [Multimodal Transformers](#multimodal-transformers)
  * [Multimodal Memory](#multimodal-memory)
* [应用与数据集（Applications and Datasets）](#applications-and-datasets)

 # 文献

## 综述

[Foundations and Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions](https://arxiv.org/abs/2209.03430), arxiv 2023

[Multimodal Learning with Transformers: A Survey](https://arxiv.org/abs/2206.06488), TPAMI 2023

[Multimodal Image Synthesis and Editing: The Generative AI Era](https://doi.org/10.1109/TPAMI.2023.3305243), TPAMI 2023

[Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods](https://doi.org/10.1613/jair.1.11688), JAIR 2021

[Experience Grounds Language](https://arxiv.org/abs/2004.10151), EMNLP 2020

[A Survey of Reinforcement Learning Informed by Natural Language](https://arxiv.org/abs/1906.03926), IJCAI 2019

[Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406), TPAMI 2019

[Multimodal Intelligence: Representation Learning, Information Fusion, and Applications](https://arxiv.org/abs/1911.03977), arXiv 2019

[Deep Multimodal Representation Learning: A Survey](https://ieeexplore.ieee.org/abstract/document/8715409), arXiv 2019

[Guest Editorial: Image and Language Understanding](https://link.springer.com/article/10.1007/s11263-017-0993-y), IJCV 2017

[Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538), TPAMI 2013

[A Survey of Socially Interactive Robots](https://www.cs.cmu.edu/~illah/PAPERS/socialroboticssurvey.pdf), 2003

## 核心领域

### 多模态表示（Multimodal Representations）

[Identifiability Results for Multimodal Contrastive Learning](https://arxiv.org/abs/2303.09166), ICLR 2023 [[code]](https://github.com/imantdaunhawer/multimodal-contrastive-learning)

[Unpaired Vision-Language Pre-training via Cross-Modal CutMix](https://arxiv.org/abs/2206.08919), ICML 2022.

[Balanced Multimodal Learning via On-the-fly Gradient Modulation](https://arxiv.org/abs/2203.15332), CVPR 2022

[Unsupervised Voice-Face Representation Learning by Cross-Modal Prototype Contrast](https://arxiv.org/abs/2204.14057), IJCAI 2021 [[code]](https://github.com/Cocoxili/CMPC)

[Towards a Unified Foundation Model: Jointly Pre-Training Transformers on Unpaired Images and Text](https://arxiv.org/abs/2112.07074), arXiv 2021

[FLAVA: A Foundational Language And Vision Alignment Model](https://arxiv.org/abs/2112.04482), arXiv 2021

[Transformer is All You Need: Multimodal Multitask Learning with a Unified Transformer](https://arxiv.org/abs/2102.10772), arXiv 2021

[MultiBench: Multiscale Benchmarks for Multimodal Representation Learning](https://arxiv.org/abs/2107.07502), NeurIPS 2021 [[code]](https://github.com/pliang279/MultiBench)

[Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206), ICML 2021 [[code]](https://github.com/deepmind/deepmind-research/tree/master/perceiver)

[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), arXiv 2021 [[blog]]([blog](https://openai.com/blog/clip/)) [[code]](https://github.com/OpenAI/CLIP)

[VinVL: Revisiting Visual Representations in Vision-Language Models](https://arxiv.org/abs/2101.00529), arXiv 2021 [[blog]](https://www.microsoft.com/en-us/research/blog/vinvl-advancing-the-state-of-the-art-for-vision-language-models/?OCID=msr_blog_VinVL_fb) [[code]](https://github.com/pzzhang/VinVL)

### 多模态融合（Multimodal Fusion）

[Robust Contrastive Learning against Noisy Views](https://arxiv.org/abs/2201.04309), arXiv 2022

[Cooperative Learning for Multi-view Analysis](https://arxiv.org/abs/2112.12337), arXiv 2022

[What Makes Multi-modal Learning Better than Single (Provably)](https://arxiv.org/abs/2106.04538), NeurIPS 2021

[Efficient Multi-Modal Fusion with Diversity Analysis](https://dl.acm.org/doi/abs/10.1145/3474085.3475188), ACMMM 2021

[Attention Bottlenecks for Multimodal Fusion](https://arxiv.org/abs/2107.00135), NeurIPS 2021

[VMLoc: Variational Fusion For Learning-Based Multimodal Camera Localization](https://arxiv.org/abs/2003.07289), AAAI 2021

[Trusted Multi-View Classification](https://openreview.net/forum?id=OOsR8BzCnl5), ICLR 2021 [[code]](https://github.com/hanmenghan/TMC)

### 多模态对齐（Multimodal Alignment）

[Reconsidering Representation Alignment for Multi-view Clustering](https://openaccess.thecvf.com/content/CVPR2021/html/Trosten_Reconsidering_Representation_Alignment_for_Multi-View_Clustering_CVPR_2021_paper.html), CVPR 2021 [[code]](https://github.com/DanielTrosten/mvc)

[CoMIR: Contrastive Multimodal Image Representation for Registration](https://arxiv.org/pdf/2006.06325.pdf), NeurIPS 2020 [[code]](https://github.com/MIDA-group/CoMIR)

### 多模态预训练（Multimodal Pretraining）
[Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651), NeurIPS 2021 Spotlight [[code]](https://github.com/salesforce/ALBEF)

[Less is More: ClipBERT for Video-and-Language Learning via Sparse Sampling](https://arxiv.org/abs/2102.06183), CVPR 2021 [[code]](https://github.com/jayleicn/ClipBERT)

[Transformer is All You Need: Multimodal Multitask Learning with a Unified Transformer](https://arxiv.org/abs/2102.10772), arXiv 2021



### 多模态翻译（Multimodal Translation）

[Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092), ICML 2021 [[code]](https://github.com/openai/DALL-E)


### 跨模态检索（Crossmodal Retrieval）

[Learning with Noisy Correspondence for Cross-modal Matching](https://proceedings.neurips.cc/paper/2021/file/f5e62af885293cf4d511ceef31e61c80-Paper.pdf), NeurIPS 2021 [[code]](https://github.com/XLearning-SCU/2021-NeurIPS-NCR)

[MURAL: Multimodal, Multitask Retrieval Across Languages](https://arxiv.org/abs/2109.05125), arXiv 2021


### 多模态共同学习（Multimodal Co-learning）

[Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918), ICML 2021

[Multimodal Co-learning: Challenges, Applications with Datasets, Recent Advances and Future Directions](https://arxiv.org/abs/2107.13782), arXiv 2021


### 模态缺失（Missing or Imperfect Modalities）

[A Variational Information Bottleneck Approach to Multi-Omics Data Integration](https://arxiv.org/abs/2102.03014), AISTATS 2021 [[code]](https://github.com/chl8856/DeepIMV)

[SMIL: Multimodal Learning with Severely Missing Modality](https://arxiv.org/abs/2103.05677), AAAI 2021



### 多模态模型分析（Analysis of Multimodal Models）

[M2Lens: Visualizing and Explaining Multimodal Models for Sentiment Analysis](https://arxiv.org/abs/2107.08264), IEEE TVCG 2022

[Decoupling the Role of Data, Attention, and Losses in Multimodal Transformers](https://arxiv.org/abs/2102.00529), TACL 2021



### 知识图谱（Knowledge Graphs and Knowledge Bases）

[MMKG: Multi-Modal Knowledge Graphs](https://arxiv.org/abs/1903.05485), ESWC 2019

[Answering Visual-Relational Queries in Web-Extracted Knowledge Graphs](https://arxiv.org/abs/1709.02314), AKBC 2019


### 可解释的学习（Intepretable Learning）

[Multimodal Explanations by Predicting Counterfactuality in Videos](https://arxiv.org/abs/1812.01263), CVPR 2019

[Multimodal Explanations: Justifying Decisions and Pointing to the Evidence](https://arxiv.org/abs/1802.08129), CVPR 2018 [[code]](https://github.com/Seth-Park/MultimodalExplanations)

[Do Explanations make VQA Models more Predictable to a Human?](https://arxiv.org/abs/1810.12366), EMNLP 2018



### 生成学习（Generative Learning）

[MMVAE+: Enhancing the Generative Quality of Multimodal VAEs without Compromises](https://openreview.net/forum?id=sdQGxouELX), ICLR 2023 [[code]](https://github.com/epalu/mmvaeplus)

[On the Limitations of Multimodal VAEs](https://arxiv.org/abs/2110.04121), ICLR 2022 [[code]](https://openreview.net/attachment?id=w-CPUXXrAj&name=supplementary_material)

[Generalized Multimodal ELBO](https://openreview.net/forum?id=5Y21V0RDBV), ICLR 2021 [[code]](https://github.com/thomassutter/MoPoE)



### 半监督学习（Semi-supervised Learning）

[Semi-supervised Vision-language Mapping via Variational Learning](https://ieeexplore.ieee.org/document/7989160), ICRA 2017

[Semi-supervised Multimodal Hashing](https://arxiv.org/abs/1712.03404), arXiv 2017



### 自监督学习（Self-supervised Learning）

[DABS: A Domain-Agnostic Benchmark for Self-Supervised Learning](https://arxiv.org/abs/2111.12062), NeurIPS 2021 Datasets & Benchmarks Track [[code]](https://github.com/alextamkin/dabs)

[Self-Supervised Learning by Cross-Modal Audio-Video Clustering](https://arxiv.org/abs/1911.12667), NeurIPS 2020 [[code]](https://github.com/HumamAlwassel/XDC)

[Self-Supervised MultiModal Versatile Networks](https://arxiv.org/abs/2006.16228), NeurIPS 2020 [[code]](https://tfhub.dev/deepmind/mmv/s3d/1)

[Labelling Unlabelled Videos from Scratch with Multi-modal Self-supervision](https://arxiv.org/abs/2006.13662), NeurIPS 2020 [[code]](https://www.robots.ox.ac.uk/~vgg/research/selavi/)



### 语言模型（Language Models）

[Neural Language Modeling with Visual Features](https://arxiv.org/abs/1903.02930), arXiv 2019

[Learning Multi-Modal Word Representation Grounded in Visual Context](https://arxiv.org/abs/1711.03483), AAAI 2018


### 对抗学习（Adversarial Attacks）

[Attend and Attack: Attention Guided Adversarial Attacks on Visual Question Answering Models](https://nips2018vigil.github.io/static/papers/accepted/33.pdf), NeurIPS Workshop on Visually Grounded Interaction and Language 2018

[Attacking Visual Language Grounding with Adversarial Examples: A Case Study on Neural Image Captioning](https://arxiv.org/abs/1712.02051), ACL 2018 [[code]](https://github.com/huanzhang12/ImageCaptioningAttack)

[Fooling Vision and Language Models Despite Localization and Attention Mechanism](https://arxiv.org/abs/1709.08693), CVPR 2018


### 少样本学习（Few-Shot Learning）

[Language to Network: Conditional Parameter Adaptation with Natural Language Descriptions](https://www.aclweb.org/anthology/2020.acl-main.625/), ACL 2020

[Shaping Visual Representations with Language for Few-shot Classification](https://arxiv.org/abs/1911.02683), ACL 2020



### 公平公正问题（Bias and Fairness）

[Worst of Both Worlds: Biases Compound in Pre-trained Vision-and-Language Models](https://arxiv.org/abs/2104.08666), arXiv 2021

[Towards Debiasing Sentence Representations](https://arxiv.org/abs/2007.08100), ACL 2020 [[code]](https://github.com/pliang279/sent_debias)

[FairCVtest Demo: Understanding Bias in Multimodal Learning with a Testbed in Fair Automatic Recruitment](https://arxiv.org/abs/2009.07025), ICMI 2020 [[code]](https://github.com/BiDAlab/FairCVtest)



### 人在回路学习（Human in the Loop Learning）

[Human in the Loop Dialogue Systems](https://sites.google.com/view/hlds-2020/home), NeurIPS 2020 workshop

[Human And Machine in-the-Loop Evaluation and Learning Strategies](https://hamlets-workshop.github.io/), NeurIPS 2020 workshop

[Human-centric dialog training via offline reinforcement learning](https://arxiv.org/abs/2010.05848), EMNLP 2020 [[code]](https://github.com/natashamjaques/neural_chat/tree/master/BatchRL)



## 架构

### Multimodal Transformers

[Pretrained Transformers As Universal Computation Engines](https://arxiv.org/abs/2103.05247), AAAI 2022

[Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206), ICML 2021

[FLAVA: A Foundational Language And Vision Alignment Model](https://arxiv.org/abs/2112.04482), arXiv 2021

[PolyViT: Co-training Vision Transformers on Images, Videos and Audio](https://arxiv.org/abs/2111.12993), arXiv 2021

[VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text](https://arxiv.org/abs/2104.11178), NeurIPS 2021 [[code]](https://github.com/google-research/google-research/tree/master/vatt)

[Parameter Efficient Multimodal Transformers for Video Representation Learning](https://arxiv.org/abs/2012.04124), ICLR 2021 [[code]](https://github.com/sangho-vision/avbert)

### Multimodal Memory

[Multimodal Transformer with Variable-length Memory for Vision-and-Language Navigation](https://arxiv.org/abs/2111.05759), arXiv 2021

[History Aware Multimodal Transformer for Vision-and-Language Navigation](https://arxiv.org/abs/2110.13309), NeurIPS 2021 [[code]](https://cshizhe.github.io/projects/vln_hamt.html)

[Episodic Memory in Lifelong Language Learning](https://arxiv.org/abs/1906.01076), NeurIPS 2019

[ICON: Interactive Conversational Memory Network for Multimodal Emotion Detection](https://aclanthology.org/D18-1280.pdf), EMNLP 2018

[Multimodal Memory Modelling for Video Captioning](https://arxiv.org/abs/1611.05592), CVPR 2018

[Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417), ICML 2016

## 应用与数据集
