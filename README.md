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

### Multimodal Alignment

[Reconsidering Representation Alignment for Multi-view Clustering](https://openaccess.thecvf.com/content/CVPR2021/html/Trosten_Reconsidering_Representation_Alignment_for_Multi-View_Clustering_CVPR_2021_paper.html), CVPR 2021 [[code]](https://github.com/DanielTrosten/mvc)

[CoMIR: Contrastive Multimodal Image Representation for Registration](https://arxiv.org/pdf/2006.06325.pdf), NeurIPS 2020 [[code]](https://github.com/MIDA-group/CoMIR)
