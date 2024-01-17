# 多模态学习调研

By Xiangyang HU(232050209@hdu.edu.cn)

## 目录
* [会议列表](#会议列表)
* [综述](#综述)
* [核心领域](#核心领域)
  * [多模态表示（Multimodal Representations）](#多模态表示multimodal-representations)
  * [多模态融合（Multimodal Fusion）](#多模态融合multimodal-fusion)
  * [多模态对齐（Multimodal Alignment）](#多模态对齐multimodal-alignment)
  * [多模态预训练（Multimodal Pretraining）](#多模态预训练multimodal-pretraining)
  * [多模态分割（Multimodal Segmentation）](#多模态分割multimodal-segmentation)
  * [多模态翻译（Multimodal Translation）](#多模态翻译multimodal-translation)
  * [跨模态检索（Crossmodal Retrieval）](#跨模态检索crossmodal-retrieval)
  * [多模态共同学习（Multimodal Co-learning）](#多模态共同学习multimodal-colearning)
  * [模态缺失（Missing or Imperfect Modalities）](#模态缺失missing-or-imperfect-modalities)
  * [多模态模型分析（Analysis of Multimodal Models）](#多模态模型分析analysis-of-multimodal-models)
  * [多模态聚类（Cluster of Multimodal Models）](#多模态聚类cluster-of-multimodal-models)
  * [多模态推理（Reason of Multimodal Models）](#多模态推理reason-of-multimodal-models)
  * [多模态情感（Multimodal Emotion）](#多模态情感multimodal-emotion)
  * [知识图谱（Knowledge Graphs and Knowledge Bases）](#知识图谱knowledge-graphs-and-knowledge-bases)
  * [可解释的学习（Intepretable Learning）](#可解释的学习intepretable-learning)
  * [生成学习（Generative Learning）](#生成学习generative-learning)
  * [半监督学习（Semi-supervised Learning）](#半监督学习semi-supervised-learning)
  * [自监督学习（Self-supervised Learning）](#自监督学习self-supervised-learning)
  * [点云（PointCloud）](#点云Point-Cloud)
  * [对比学习（Contrastive Learning）](#对比学习contrastive-learning)
  * [语言模型（Language Models）](#语言模型language-models)
  * [对抗学习（Adversarial Attacks）](#对抗学习adversarial-attacks)
  * [少样本学习（Few-Shot Learning）](#少样本学习few-shot-learning)
  * [公平公正问题（Bias and Fairness）](#公平公正问题bias-and-fairness)
  * [人在回路学习（Human in the Loop Learning）](#人在回路学习human-in-the-loop-learning)
* [架构](#架构)
  * [Multimodal Transformers](#multimodal-transformers)
  * [Multimodal Memory](#multimodal-memory)
* [应用与数据集（Applications and Datasets）](#applications-and-datasets)
  * [医学（Medicine）](#医学Medicine)
  * [识别（Recognition）](#识别Recognition)
  * [检测（Detection）](#检测Detection)
  * [预测（Prediction）](#预测Prediction)
  * [推荐算法（Recommendation）](#推荐算法Recommendation)
  * [数据集（Dataset）](#数据集Dataset)

 # 会议列表

 | 序号| 会议名 | 注册时间 | 提交截止时间 | 第一次回复时间 | 备注 |
 | :---: | :---: | :----- | :----- | :----- | :----- |
 | **1** | **** |||||
 | **2** | **** |||||
 | **3** | **** |||||
 | **4** | **** |||||
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

### 医学多模态综述

[Multimodal biomedical AI](https://doi.org/10.1038/s41591-022-01981-2), 2022 Nature medicine

[A Review of the Application of Multi-modal Deep Learning in Medicine: Bibliometrics and Future Directions](https://link.springer.com/article/10.1007/s44196-023-00225-6), 2023 International Journal of Computationallntelligence Systems

[Survey on deep learning in multimodal medical imaging for cancer detection](https://doi.org/10.1007/s00521-023-09214-4), 2023 Neural Computing and Applications

[A review on multimodal medical image fusion: Compendious analysis of  medical modalities, multimodal databases, fusion techniques and  quality metrics](https://doi.org/10.1016/j.compbiomed.2022.105253), 2022 Computers in Biology and Medicine

[Integration of deep learning-based image analysis and genomic data in cancer pathology: A systematic review](https://doi.org/10.1016/j.ejca.2021.10.007), 2022 ScienceDirect

[医学图像融合方法综述](http://www.cjig.cn/html/2023/1/20230107.htm), 2023 中国图像图形学报

[A review on multimodal machine learning in medical diagnostics](http://dx.doi.org/10.3934/mbe.2023382), 2023 MBE

[Beyond Medical Imaging: A Review of Multimodal Deep Learning in Radiology](https://www.techrxiv.org/users/684383/articles/678856-beyond-medical-imaging-a-review-of-multimodal-deep-learning-in-radiology), 2023 techrxiv

[An overview of deep learning methods for multimodal medical data mining](https://doi.org/10.1016/j.eswa.2022.117006), 2022 Expert Systems With Applications

[Deep learning in multimodal medical imaging for cancer detection](https://doi.org/10.1007/s00521-023-08955-6), 2022 Neural Computing and Applications

[Machine Learning in Multimodal Medical Imaging](https://doi.org/10.1155/2017/1278329), 2017 BioMed Research International


### 图像融合综述

[Multispectral and hyperspectral image fusion in remote sensing: A survey Gemine Vivone](https://doi.org/10.1016/j.inffus.2022.08.032), 2023 Information Fusion

[Current advances and future perspectives of image fusion: A comprehensive review](https://doi.org/10.1016/j.inffus.2022.09.019), 2023 Information Fusion


## 核心领域

### 多模态表示（Multimodal Representations）

[Decoding Visual Neural Representations by Multimodal Learning of Brain-Visual-Linguistic Features](https://doi.org/10.1109/TPAMI.2023.3263181), TPAMI 2023 [[code]](https://github.com/ChangdeDu/BraVL)

[Universal Multimodal Representation for Language Understanding](https://doi.org/10.1109/TPAMI.2023.3234170), TPAMI 2023

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

[IFSepR: A General Framework for Image Fusion Based on Separate Representation Learning](https://doi.org/10.1109/TMM.2021.3129354), TMM 2023

[F-DARTS: Foveated Differentiable Architecture Search Based Multimodal Medical Image Fusion](https://doi.org/10.1109/TMI.2023.3283517), TMI 2023 [[code]](https://github.com/VictorWylde/F-DARTS)

[MATR: Multimodal Medical Image Fusion via Multiscale Adaptive Transformer](https://doi.org/10.1109/TIP.2022.3193288), TIP 2023 [[code]](https://github.com/tthinking/MATR)

[Dif-Fusion: Toward High Color Fidelity in Infrared and Visible Image Fusion With Diffusion Models](https://github.com/GeoVectorMatrix/Dif-Fusion), TIP 2023 [[code]](https://doi.org/10.48550/arXiv.2301.08072)

[Channel Exchanging Networks for Multimodal and Multitask Dense Image Prediction](https://doi.org/10.1109/TPAMI.2022.3211086), TPAMI 2023 [[code]](https://github.com/yikaiw/CEN)

[Dynamic Multimodal Fusion](https://openaccess.thecvf.com/content/CVPR2023W/MULA/html/Xue_Dynamic_Multimodal_Fusion_CVPRW_2023_paper.html), CVPRW 2023 [[code]](https://github.com/zihuixue/ DynMM)

[Robust Contrastive Learning against Noisy Views](https://arxiv.org/abs/2201.04309), arXiv 2022

[Cooperative Learning for Multi-view Analysis](https://arxiv.org/abs/2112.12337), arXiv 2022

[What Makes Multi-modal Learning Better than Single (Provably)](https://arxiv.org/abs/2106.04538), NeurIPS 2021

[Efficient Multi-Modal Fusion with Diversity Analysis](https://dl.acm.org/doi/abs/10.1145/3474085.3475188), ACMMM 2021

[Attention Bottlenecks for Multimodal Fusion](https://arxiv.org/abs/2107.00135), NeurIPS 2021

[VMLoc: Variational Fusion For Learning-Based Multimodal Camera Localization](https://arxiv.org/abs/2003.07289), AAAI 2021

[Trusted Multi-View Classification](https://openreview.net/forum?id=OOsR8BzCnl5), ICLR 2021 [[code]](https://github.com/hanmenghan/TMC)

### 多模态对齐（Multimodal Alignment）

[Interpretable Multi-Modal Image Registration Network Based on Disentangled Convolutional Sparse Coding](https://doi.org/10.1109/TIP.2023.3240024), TIP 2023 [[code]](https://github.com/lep990816/Interpretable- Multi-modal-Image-Registration)

[OSAN: A One-Stage Alignment Network to Unify Multimodal Alignment and Unsupervised Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_OSAN_A_One-Stage_Alignment_Network_To_Unify_Multimodal_Alignment_and_CVPR_2023_paper.html), CVPR 2023

[MEGA: Multimodal Alignment Aggregation and Distillation For Cinematic Video Segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Sadoughi_MEGA_Multimodal_Alignment_Aggregation_and_Distillation_For_Cinematic_Video_Segmentation_ICCV_2023_paper.html), ICCV 2023 [[code]](https://github.com/ppapalampidi/GraphTP)

[Reconsidering Representation Alignment for Multi-view Clustering](https://openaccess.thecvf.com/content/CVPR2021/html/Trosten_Reconsidering_Representation_Alignment_for_Multi-View_Clustering_CVPR_2021_paper.html), CVPR 2021 [[code]](https://github.com/DanielTrosten/mvc)

[CoMIR: Contrastive Multimodal Image Representation for Registration](https://arxiv.org/pdf/2006.06325.pdf), NeurIPS 2020 [[code]](https://github.com/MIDA-group/CoMIR)

### 多模态预训练（Multimodal Pretraining）
[End-to-End Pre-Training With Hierarchical Matching and Momentum Contrast for Text-Video Retrieval](https://doi.org/10.1109/TIP.2023.3275071), TIP 2023 [[code]](https://github.com/cheetah003/HMMC)

[REVEAL: Retrieval-Augmented Visual-Language Pre-Training with Multi-Source Multimodal Knowledge Memory](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_REVEAL_Retrieval-Augmented_Visual-Language_Pre-Training_With_Multi-Source_Multimodal_Knowledge_Memory_CVPR_2023_paper.html), CVPR 2023 [[code]](https://github.com/ReVeaL-CVPR/ReVeaL-CVPR.github.io)

[Enhancing Sentence Representation with Visually-supervised Multimodal Pre-training](https://doi.org/10.1145/3581783.3612254), ACM MM 2023 [[code]](https://github.com/gentlefress/ViP)

[Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651), NeurIPS 2021 Spotlight [[code]](https://github.com/salesforce/ALBEF)

[Less is More: ClipBERT for Video-and-Language Learning via Sparse Sampling](https://arxiv.org/abs/2102.06183), CVPR 2021 [[code]](https://github.com/jayleicn/ClipBERT)

[Transformer is All You Need: Multimodal Multitask Learning with a Unified Transformer](https://arxiv.org/abs/2102.10772), arXiv 2021

### 多模态分割（Multimodal Segmentation）

[Multimodal Variational Auto-encoder based Audio-Visual Segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Multimodal_Variational_Auto-encoder_based_Audio-Visual_Segmentation_ICCV_2023_paper.html), ICCV 2023 [[code]](https://github.com/OpenNLPLab/MMVAE-AVS)

[MEGA: Multimodal Alignment Aggregation and Distillation For Cinematic Video Segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Sadoughi_MEGA_Multimodal_Alignment_Aggregation_and_Distillation_For_Cinematic_Video_Segmentation_ICCV_2023_paper.html), ICCV 2023 [[code]](https://github.com/ppapalampidi/GraphTP)

[M3AE: Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities](https://arxiv.org/abs/2303.05302), AAAI 2023 [[code]](https://github.com/ccarliu/m3ae)

### 多模态翻译（Multimodal Translation）

[CLIPTrans: Transferring Visual Knowledge with Pre-trained Models for Multimodal Machine Translation](https://openaccess.thecvf.com/content/ICCV2023/html/Gupta_CLIPTrans_Transferring_Visual_Knowledge_with_Pre-trained_Models_for_Multimodal_Machine_ICCV_2023_paper.html), ICCV 2023 [[code]](https://github.com/devaansh100/CLIPTrans)

[Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092), ICML 2021 [[code]](https://github.com/openai/DALL-E)


### 跨模态检索（Crossmodal Retrieval）

[End-to-End Pre-Training With Hierarchical Matching and Momentum Contrast for Text-Video Retrieval](https://doi.org/10.1109/TIP.2023.3275071), TIP 2023 [[code]](https://github.com/cheetah003/HMMC)

[Efficient Token-Guided Image-Text Retrieval With Consistent Multimodal Contrastive Training](https://github.com/LCFractal/TGDT), TIP 2023 [[code]](https://doi.org/10.1109/TIP.2023.3286710)

[Learning with Noisy Correspondence for Cross-modal Matching](https://proceedings.neurips.cc/paper/2021/file/f5e62af885293cf4d511ceef31e61c80-Paper.pdf), NeurIPS 2021 [[code]](https://github.com/XLearning-SCU/2021-NeurIPS-NCR)

[MURAL: Multimodal, Multitask Retrieval Across Languages](https://arxiv.org/abs/2109.05125), arXiv 2021


### 多模态共同学习（Multimodal Co-learning）

[Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918), ICML 2021

[Multimodal Co-learning: Challenges, Applications with Datasets, Recent Advances and Future Directions](https://arxiv.org/abs/2107.13782), arXiv 2021


### 模态缺失（Missing or Imperfect Modalities）

[AutoEncoder-Driven Multimodal Collaborative Learning for Medical Image Synthesis](https://doi.org/10.1007/s11263-023-01791-0), IJCV 2023 [[code]](https://github.com/bcaosudo/AE-GAN)

[GCNet: Graph Completion Network for Incomplete Multimodal Learning in Conversation](https://doi.org/10.1109/TPAMI.2023.3234553), TPAMI 2023 [[code]](https://github.com/zeroQiaoba/GCNet)

[Multimodal Prompting with Missing Modalities for Visual Recognition](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Multimodal_Prompting_With_Missing_Modalities_for_Visual_Recognition_CVPR_2023_paper.html), CVPR 2023 

[M3AE: Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities](https://arxiv.org/abs/2303.05302), AAAI 2023 [[code]](https://github.com/ccarliu/m3ae)

[A Variational Information Bottleneck Approach to Multi-Omics Data Integration](https://arxiv.org/abs/2102.03014), AISTATS 2021 [[code]](https://github.com/chl8856/DeepIMV)

[SMIL: Multimodal Learning with Severely Missing Modality](https://arxiv.org/abs/2103.05677), AAAI 2021


### 多模态模型分析（Analysis of Multimodal Models）

[M2Lens: Visualizing and Explaining Multimodal Models for Sentiment Analysis](https://arxiv.org/abs/2107.08264), IEEE TVCG 2022

[Decoupling the Role of Data, Attention, and Losses in Multimodal Transformers](https://arxiv.org/abs/2102.00529), TACL 2021

### 多模态聚类（Cluster of Multimodal Models）

[Graph Embedding Contrastive Multi-Modal Representation Learning for Clustering](https://doi.org/10.1109/TIP.2023.3240863), TIP 2023 [[code]](https://github.com/xdweixia/GECMC)

### 多模态推理（Reason of Multimodal Models）

[Experts Collaboration Learning for Continual Multi-Modal Reasoning](https://doi.org/10.1109/TIP.2023.3310336), TIP 2023

### 多模态情感（Multimodal Emotion）

[Sensing Micro-Motion Human Patterns using Multimodal mmRadar and Video Signal for Affective and Psychological Intelligence](https://doi.org/10.1145/3581783.3611754), ACM MM 2023 [[code]](https://remap-dataset.github.io/ReMAP)

[Multimodal Adaptive Emotion Transformer with Flexible Modality Inputs on A Novel Dataset with Continuous Labels](https://doi.org/10.1145/3581783.3613797), ACM MM 2023 [[code]](https://github.com/935963004/MAET)

[MEDIC: A Multimodal Empathy Dataset in Counseling](https://doi.org/10.1145/3581783.3612346), ACM MM 2023 [[code]](https://ustc-ac.github.io/datasets/medic/)

[Multi-label Emotion Analysis in Conversation via Multimodal Knowledge Distillation](https://doi.org/10.1145/3581783.3612517), ACM MM 2023 [[code]](https://github.com/devulapa/multimodal-emotion-recognition)

[General Debiasing for Multimodal Sentiment Analysis](https://doi.org/10.1145/3581783.3612051), ACM MM 2023 [[code]](https://github.com/Teng-Sun/GEAR)

[Mining High-quality Samples from Raw Data and Majority Voting Method for Multimodal Emotion Recognition](https://doi.org/10.1145/3581783.3612862), ACM MM 2023 

[Graph to Grid: Learning Deep Representations for Multimodal Emotion Recognition](https://doi.org/10.1145/3581783.3612074), ACM MM 2023 [[code]](https://github.com/Jinminbox/G2G)

[Few-shot Multimodal Sentiment Analysis Based on Multimodal Probabilistic Fusion Prompts](https://doi.org/10.1145/3581783.3612181), ACM MM 2023 [[code]](https://github.com/YangXiaocui1215/ MultiPoint)

[AcFormer: An Aligned and Compact Transformer for Multimodal Sentiment Analysis](https://doi.org/10.1145/3581783.3611974), ACM MM 2023 [[code]](https://github.com/dingchaoyue/AcFormer)

[Building Robust Multimodal Sentiment Recognition via a Simple yet Effective Multimodal Transformer](https://doi.org/10.1145/3581783.3612872), ACM MM 2023 [[code]](https://github.com/dingchaoyue/Multimodal- Emotion-Recognition-MER-and-MuSe-2023-Challenges)

### 知识图谱（Knowledge Graphs and Knowledge Bases）

[Joint Multimodal Entity-Relation Extraction Based on Edge-Enhanced Graph Alignment Network and Word-Pair Relation Tagging](https://ojs.aaai.org/index.php/AAAI/article/view/26309), AAAI 2023 [[code]](https://github.com/YuanLi95/EEGA-for-JMERE)

[TIVA-KG: A Multimodal Knowledge Graph with Text, Image, Video and Audio](https://doi.org/10.1145/3581783.3612266), ACM MM 2023 [[code]](http://mn.cs.tsinghua.edu.cn/tivakg)

[MMKG: Multi-Modal Knowledge Graphs](https://arxiv.org/abs/1903.05485), ESWC 2019

[Answering Visual-Relational Queries in Web-Extracted Knowledge Graphs](https://arxiv.org/abs/1709.02314), AKBC 2019

### 可解释的学习（Intepretable Learning）

[Multimodal Explanations by Predicting Counterfactuality in Videos](https://arxiv.org/abs/1812.01263), CVPR 2019

[Multimodal Explanations: Justifying Decisions and Pointing to the Evidence](https://arxiv.org/abs/1802.08129), CVPR 2018 [[code]](https://github.com/Seth-Park/MultimodalExplanations)

[Do Explanations make VQA Models more Predictable to a Human?](https://arxiv.org/abs/1810.12366), EMNLP 2018



### 生成学习（Generative Learning）

[MMVAE+: Enhancing the Generative Quality of Multimodal VAEs without Compromises](https://openreview.net/forum?id=sdQGxouELX), ICLR 2023 [[code]](https://github.com/epalu/mmvaeplus)

[Text-to-Image Diffusion Models can be Easily Backdoored through Multimodal Data Poisoning](https://doi.org/10.1145/3581783.3612108), ACM MM 2023 [[code]](https://github.com/sf-zhai/BadT2I)

[On the Limitations of Multimodal VAEs](https://arxiv.org/abs/2110.04121), ICLR 2022 [[code]](https://openreview.net/attachment?id=w-CPUXXrAj&name=supplementary_material)

[Generalized Multimodal ELBO](https://openreview.net/forum?id=5Y21V0RDBV), ICLR 2021 [[code]](https://github.com/thomassutter/MoPoE)



### 半监督学习（Semi-supervised Learning）

[Supervised Phenotype Discovery From Multimodal Brain Imaging](https://doi.org/10.1109/TMI.2022.3218720), TMI 2023 [[code]](https://github.com/ weikanggong/SuperBigFLICA)

[Harvard Glaucoma Detection and Progression: A Multimodal Multitask Dataset and Generalization-Reinforced Semi-Supervised Learning](https://openaccess.thecvf.com/content/ICCV2023/html/Luo_Harvard_Glaucoma_Detection_and_Progression_A_Multimodal_Multitask_Dataset_and_ICCV_2023_paper.html), ICCV 2023 [[code]](https://ophai.hms.harvard.edu/ datasets/harvard-gdp1000)

[Semi-supervised Vision-language Mapping via Variational Learning](https://ieeexplore.ieee.org/document/7989160), ICRA 2017

[Semi-supervised Multimodal Hashing](https://arxiv.org/abs/1712.03404), arXiv 2017



### 自监督学习（Self-supervised Learning）

[Self-Supervised Learning for Multimodal Non-Rigid 3D Shape Matching](https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Self-Supervised_Learning_for_Multimodal_Non-Rigid_3D_Shape_Matching_CVPR_2023_paper.html), CVPR 2023 [[code]](https://github.com/ dongliangcao/Self-Supervised-Multimodal-Shape-Matching)

[Best of Both Worlds: Multimodal Contrastive Learning with Tabular and Imaging Data](https://openaccess.thecvf.com/content/CVPR2023/html/Hager_Best_of_Both_Worlds_Multimodal_Contrastive_Learning_With_Tabular_and_CVPR_2023_paper.html), CVPR 2023 [[code]](https : //github . com / paulhager / MMCL - Tabular - Imaging)

[DABS: A Domain-Agnostic Benchmark for Self-Supervised Learning](https://arxiv.org/abs/2111.12062), NeurIPS 2021 Datasets & Benchmarks Track [[code]](https://github.com/alextamkin/dabs)

[Self-Supervised Learning by Cross-Modal Audio-Video Clustering](https://arxiv.org/abs/1911.12667), NeurIPS 2020 [[code]](https://github.com/HumamAlwassel/XDC)

[Self-Supervised MultiModal Versatile Networks](https://arxiv.org/abs/2006.16228), NeurIPS 2020 [[code]](https://tfhub.dev/deepmind/mmv/s3d/1)

[Labelling Unlabelled Videos from Scratch with Multi-modal Self-supervision](https://arxiv.org/abs/2006.13662), NeurIPS 2020 [[code]](https://www.robots.ox.ac.uk/~vgg/research/selavi/)

### 对比学习（Contrastive Learning）

[Revisiting Multimodal Representation in Contrastive Learning: From Patch and Token Embeddings to Finite Discrete Tokens](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Revisiting_Multimodal_Representation_in_Contrastive_Learning_From_Patch_and_Token_CVPR_2023_paper.html), CVPR 2023 [[code]](https://github.com/yuxiaochen1103/FDT)

[Best of Both Worlds: Multimodal Contrastive Learning with Tabular and Imaging Data](https://openaccess.thecvf.com/content/CVPR2023/html/Hager_Best_of_Both_Worlds_Multimodal_Contrastive_Learning_With_Tabular_and_CVPR_2023_paper.html), CVPR 2023 [[code]](https://github.com/paulhager/MMCL-Tabular-Imaging)

[AdvCLIP: Downstream-agnostic Adversarial Examples in Multimodal Contrastive Learning](https://doi.org/10.1145/3581783.3612454), ACM MM 2023 [[code]](https://github.com/CGCL-codes/AdvCLIP)

[Cross-modal Contrastive Learning for Multimodal Fake News Detection](https://dl.acm.org/doi/10.1145/3581783.3613850), ACM MM 2023 [[code]](https://github.com/wishever/COOLANT)

### 点云（Point Cloud）

[3D Spatial Multimodal Knowledge Accumulation for Scene Graph Prediction in Point Cloud](https://openaccess.thecvf.com/content/CVPR2023/html/Feng_3D_Spatial_Multimodal_Knowledge_Accumulation_for_Scene_Graph_Prediction_in_CVPR_2023_paper.html), CVPR 2023 [[code]](https://github.com/HHrEtvP/SMKA)

[RPEFlow: Multimodal Fusion of RGB-PointCloud-Event for Joint Optical Flow and Scene Flow Estimation](https://openaccess.thecvf.com/content/ICCV2023/html/Wan_RPEFlow_Multimodal_Fusion_of_RGB-PointCloud-Event_for_Joint_Optical_Flow_and_ICCV_2023_paper.html), ICCV 2023 [[code]](https://npucvr.github.io/RPEFlow)

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

### 医学（Medicine）

[Simultaneously-Collected Multimodal Lying Pose Dataset: Enabling In-Bed Human Pose Monitoring](https://doi.org/10.1109/TPAMI.2022.3155712), TPAMI 2023

[Graph Transformer Geometric Learning of Brain Networks Using Multimodal MR Images for Brain Age Estimation](https://doi.org/10.1109/TMI.2022.3222093), TMI 2023 [[code]](https://github. com/Hongjie97GraphTransfomer_BrainAge)

[Multimodal Transformer for Accelerated MR Imaging](https://doi.org/10.1109/TMI.2022.3180228), TMI 2023 [[code]](https://github.com/chunmeifeng/MTrans)

[Hybrid Graph Convolutional Network With Online Masked Autoencoder for Robust Multimodal Cancer Survival Prediction](https://doi.org/10.1109/TMI.2023.3253760), TMI 2023 [[code]](https://github.com/lin- lcx/HGCN)

[F-DARTS: Foveated Differentiable Architecture Search Based Multimodal Medical Image Fusion](https://doi.org/10.1109/TMI.2023.3283517), TMI 2023 [[code]](https://github.com/VictorWylde/F-DARTS)

[Supervised Phenotype Discovery From Multimodal Brain Imaging](https://doi.org/10.1109/TMI.2022.3218720), TMI 2023 [[code]](https://github.com/ weikanggong/SuperBigFLICA)

[Survival Prediction via Hierarchical Multimodal Co-Attention Transformer: A Computational Histology-Radiology Solution](https://doi.org/10.1109/TMI.2023.3263010), TMI 2023 

[GMRLNet: A Graph-Based Manifold Regularization Learning Framework for Placental Insufficiency Diagnosis on Incomplete Multimodal Ultrasound Data](https://doi.org/10.1109/TMI.2023.3278259), TMI 2023 

[Breath-Hold CBCT-Guided CBCT-to-CT Synthesis via Multimodal Unsupervised Representation Disentanglement Learning](https://doi.org/10.1109/TMI.2023.3247759), TMI 2023 

[MATR: Multimodal Medical Image Fusion via Multiscale Adaptive Transformer](https://doi.org/10.1109/TIP.2022.3193288), TIP 2023 [[code]](https://github.com/tthinking/MATR)

[Volumetric Model Genesis in Medical Domain for the Analysis of Multimodality 2-D/3-D Data Based on the Aggregation of Multilevel Features](https://doi.org/10.1109/TII.2023.3252541), TII 2023 

[Integrating Medical Domain Knowledge for Early Diagnosis of Fever of Unknown Origin: An Interpretable Hierarchical Multimodal Neural Network Approach](https://doi.org/10.1109/JBHI.2023.3306041), JBHI 2023 [[code]](https://github.com/Yovosss/iHMNNF)

[Predicting 30-Day All-Cause Hospital Readmission Using Multimodal Spatiotemporal Graph Neural Networks](https://doi.org/10.1109/JBHI.2023.3236888), JBHI 2023 [[code]](https://github.com/tsy935/readmit-stgnn, https://github.com/stanfordmlgroup/MoCo-CXR)

[Multimodal Data Matters: Language Model Pre-Training Over Structured and Unstructured Electronic Health Records](https://doi.org/10.1109/JBHI.2022.3217810), JBHI 2023 

[Real-Time Prediction for Neonatal Endotracheal Intubation Using Multimodal Transformer Network](https://doi.org/10.1109/JBHI.2023.3267521), JBHI 2023 

[Coco-Attention for Tumor Segmentation in Weakly Paired Multimodal MRI Images](https://doi.org/10.1109/JBHI.2023.3262548), JBHI 2023 

[Multimodal Fusion Network for Detecting Hyperplastic Parathyroid Glands in SPECT/CT Images](https://doi.org/10.1109/JBHI.2022.3228603), JBHI 2023 

[The Individualized Prediction of Neurocognitive Function in People Living With HIV Based on Clinical and Multimodal Connectome Data](https://doi.org/10.1109/JBHI.2023.3240508), JBHI 2023 

[AutoEncoder-Driven Multimodal Collaborative Learning for Medical Image Synthesis](https://doi.org/10.1007/s11263-023-01791-0), IJCV 2023 [[code]](https://github.com/bcaosudo/AE-GAN)

[Harvard Glaucoma Detection and Progression: A Multimodal Multitask Dataset and Generalization-Reinforced Semi-Supervised Learning](https://openaccess.thecvf.com/content/ICCV2023/html/Luo_Harvard_Glaucoma_Detection_and_Progression_A_Multimodal_Multitask_Dataset_and_ICCV_2023_paper.html), ICCV 2023 [[code]](https://ophai.hms.harvard.edu/ datasets/harvard-gdp1000)

[M3AE: Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities](https://arxiv.org/abs/2303.05302), AAAI 2023 [[code]](https://github.com/ccarliu/m3ae)

### 识别(Recognition)

#### 动作识别

[MMNet: A Model-Based Multimodal Network for Human Action Recognition in RGB-D Videos](https://doi.org/10.1109/TPAMI.2022.3177813), TPAMI 2023

[B2C-AFM: Bi-Directional Co-Temporal and Cross-Spatial Attention Fusion Model for Human Action Recognition](https://doi.org/10.1109/TIP.2023.3308750), TIP 2023 [[code]](https://github.com/gftww/B2C.git)

[Cross-scale cascade transformer for multimodal human action recognition](https://doi.org/10.1016/j.patrec.2023.02.024), PRL 2023

[Multimodal Distillation for Egocentric Action Recognition](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Multimodal_Variational_Auto-encoder_based_Audio-Visual_Segmentation_ICCV_2023_paper.html), ICCV 2023 [[code]](https://github.com/gorjanradevski/multimodal-distillation)

#### 情感识别

[Multi-Channel Weight-Sharing Autoencoder Based on Cascade Multi-Head Attention for Multimodal Emotion Recognition](https://doi.org/10.1109/TMM.2022.3144885), TMM 2023

[Multimodal Continuous Emotion Recognition: A Technical Report for ABAW5](https://openaccess.thecvf.com/content/CVPR2023W/ABAW/html/Zhang_Multimodal_Continuous_Emotion_Recognition_A_Technical_Report_for_ABAW5_CVPRW_2023_paper.html), CVPR 2023 [[code]](https://github.com/sucv/ABAW3)

[Decoupled Multimodal Distilling for Emotion Recognition](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Decoupled_Multimodal_Distilling_for_Emotion_Recognition_CVPR_2023_paper.html), CVPR 2023 [[code]](https://github.com/mdswyz/DMD)

[Revisiting Disentanglement and Fusion on Modality and Context in Conversational Multimodal Emotion Recognition](https://doi.org/10.1145/3581783.3612053), ACM MM 2023 [[code]](https://github.com/something678/TodKat)

[Mining High-quality Samples from Raw Data and Majority Voting Method for Multimodal Emotion Recognition](https://doi.org/10.1145/3581783.3612862), ACM MM 2023 

[Graph to Grid: Learning Deep Representations for Multimodal Emotion Recognition](https: //doi.org/10.1145/3581783.3612074), ACM MM 2023 [[code]](https://github.com/Jinminbox/G2G)

[Building Robust Multimodal Sentiment Recognition via a Simple yet Effective Multimodal Transformer](https://doi.org/10.1145/3581783.3612872), ACM MM 2023 [[code]](https://github.com/dingchaoyue/Multimodal- Emotion-Recognition-MER-and-MuSe-2023-Challenges)


#### 生物识别
[Learning Sparse and Discriminative Multimodal Feature Codes for Finger Recognition](https://doi.org/10.1109/TMM.2022.3144885), TMM 2023 [[code]](https://doi.org/10.1109/TMM.2021.3132166)

#### 多模态命名实体识别（MNER）

[Fine-Grained Multimodal Named Entity Recognition and Grounding with a Generative Framework](https://doi.org/10.1145/3581783.3612322), ACM MM 2023 [[code]](https://github.com/NUSTM/FMNERG.)

[MCG-MNER: A Multi-Granularity Cross-Modality Generative Framework for Multimodal NER with Instruction](https://doi.org/10.1145/3581783.3612470), ACM MM 2023 [[code]](https://github.com/jetwu-create/MCG-MNER)

### 检测（Detection）

[ReDFeat: Recoupling Detection and Description for Multimodal Feature Learning](https://doi.org/10.1109/TIP.2022.3231135), TIP 2023 [[code]](https://github.com/ACuOoOoO/ ReDFeat)

[HiDAnet: RGB-D Salient Object Detection via Hierarchical Depth Awareness](https://doi.org/10.1109/TIP.2023.3263111), TIP 2023 [[code]](https://github.com/Zongwei97/HIDANet/)

[Weakly Aligned Multimodal Flame Detection for Fire-Fighting Robots](https://doi.org/10.1109/TII.2022.3158668), TII 2023 

[Virtual Sparse Convolution for Multimodal 3D Object Detection](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Virtual_Sparse_Convolution_for_Multimodal_3D_Object_Detection_CVPR_2023_paper.html), CVPR 2023 [[code]](https://github.com/hailanyi/VirConv)

[Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection](https://openaccess.thecvf.com/content/ICCV2023/html/Flaborea_Multimodal_Motion_Conditioned_Diffusion_Model_for_Skeleton-based_Video_Anomaly_Detection_ICCV_2023_paper.html), ICCV 2023 [[code]](https://github.com/aleflabo/MoCoDAD)

### 预测（Prediction）

[Micro-Video Popularity Prediction Via Multimodal Variational Information Bottleneck](https://doi.org/10.1109/TMM.2021.3120537), TMM 2023 [[code]](https://github.com/JennyXieJiayi/HMMVED)

[Causal Conditional Hidden Markov Model for Multimodal Traffic Prediction](https://arxiv.org/abs/2301.08249), AAAI 2023 [[code]](https://github.com/EternityZY/CCHMM)

#### 生存预测

[Hybrid Graph Convolutional Network With Online Masked Autoencoder for Robust Multimodal Cancer Survival Prediction](https://doi.org/10.1109/TMI.2023.3253760), TMI 2023 [[code]](https://github.com/lin- lcx/HGCN)

[Survival Prediction via Hierarchical Multimodal Co-Attention Transformer: A Computational Histology-Radiology Solution](https://doi.org/10.1109/TMI.2023.3263010), TMI 2023 

[Multimodal Optimal Transport-based Co-Attention Transformer with Global Structure Consistency for Survival Prediction](https://arxiv.org/abs/2306.08330), ICCV 2023 [[code]](https://github.com/Innse/MOTCat)

### 推荐算法（Recommendation）

[DualGNN: Dual Graph Neural Network for Multimedia Recommendation](https://doi.org/10.1109/TMM.2021.3138298), TMM 2023 [[code]](https://github.com/wqf321/dualgnn)

[Semantic-Guided Feature Distillation for Multimodal Recommendation](https://doi.org/10.1145/3581783.3611886), ACM MM 2023 [[code]](https://github.com/HuilinChenJN/SGFD)

[A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation](https://doi.org/10.1145/3581783.3611943), ACM MM 2023 [[code]](https://github.com/enoche/FREEDOM)

[Ducho: A Unified Framework for the Extraction of Multimodal Features in Recommendation](https://doi.org/10.1145/ 3581783.3613458), ACM MM 2023 [[code]](https://github.com/sisinflab/Ducho)


### 生成（Recommendation）

[Topic-aware video summarization using multimodal transformer](https://doi.org/10.1016/j.patcog.2023.109578), PR 2023

[GraphRevisedIE: Multimodal information extraction with graph-revised network](https://doi.org/10.1016/j.patcog.2023.109542), PR 2023 [[code]](https://github. com/caop-kie/GraphRevisedIE)

[Crisis event summary generative model based on hierarchical multimodal fusion](https://doi.org/10.1016/j.patcog.2023.109890), PR 2023 

[Sentimental Visual Captioning using Multimodal Transformer](https://doi.org/10.1007/s11263-023-01752-7), IJCV 2023 [[code]](https:// github.com/ezeli/InSentiCap_ext)

[SDFusion: Multimodal 3D Shape Completion, Reconstruction, and Generation](https://openaccess.thecvf.com/content/CVPR2023/html/Cheng_SDFusion_Multimodal_3D_Shape_Completion_Reconstruction_and_Generation_CVPR_2023_paper.html), CVPR 2023 [[code]](https://yccyenchicheng.github.io/SDFusion/)

### 数据集（Dataset）

[Animal Pose Tracking: 3D Multimodal Dataset and Token-based Pose Optimization](https://doi.org/10.1007/s11263-022-01714-5), IJCV 2023 

[The MONET dataset: Multimodal drone thermal dataset recorded in rural scenarios](https://openaccess.thecvf.com/content/CVPR2023W/MULA/html/Riz_The_MONET_Dataset_Multimodal_Drone_Thermal_Dataset_Recorded_in_Rural_CVPRW_2023_paper.html), CVPRW 2023 [[code]](https: //github.com/fabiopoiesi/monet_dataset)

[Lecture Presentations Multimodal Dataset: Towards Understanding Multimodality in Educational Videos](https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Lecture_Presentations_Multimodal_Dataset_Towards_Understanding_Multimodality_in_Educational_Videos_ICCV_2023_paper.html), ICCV 2023 [[code]](https://github.com/dondongwon/LPMDataset)

[Zenseact Open Dataset: A large-scale and diverse multimodal dataset for autonomous driving](https://openaccess.thecvf.com/content/ICCV2023/html/Alibeigi_Zenseact_Open_Dataset_A_Large-Scale_and_Diverse_Multimodal_Dataset_for_ICCV_2023_paper.html), ICCV 2023 [[code]](https://zod.zenseact.com/)

[Decoding the Underlying Meaning of Multimodal Hateful Memes](https://arxiv.org/abs/2305.17678), IJCAI 2023 [[code]](https://github.com/Social-AI-Studio/HatRed)

[FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms](https://ojs.aaai.org/index.php/AAAI/article/view/26689), AAAI 2023 [[code]](https://github.com/ICTMCG/FakeSV)

[Multimodal Adaptive Emotion Transformer with Flexible Modality Inputs on A Novel Dataset with Continuous Labels](https://doi.org/10.1145/3581783.3613797), ACM MM 2023 [[code]](https://github.com/935963004/MAET)

[MEDIC: A Multimodal Empathy Dataset in Counseling](https://doi.org/10.1145/3581783.3612346), ACM MM 2023 [[code]](https://ustc-ac.github.io/datasets/medic/)

[ChinaOpen: A Dataset for Open-world Multimodal Learning](https://doi.org/10.1145/3581783.3612156), ACM MM 2023 [[code]](https://ruc-aimc-lab.github.io/ChinaOpen/)
