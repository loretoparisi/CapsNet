# CapsNet
CapsNet (Capsules Net) in Geoffrey E Hinton paper "Dynamic Routing Between Capsules"
- [What's New](https://github.com/loretoparisi/CapsNet#whats-new)
# Table of Contents
- [What's New](https://github.com/loretoparisi/CapsNet#whats-new)
- [Abstract](https://github.com/loretoparisi/CapsNet#abstract) :new:
- [Documentation](https://github.com/loretoparisi/CapsNet#documentation)
- [Discussions Groups](https://github.com/loretoparisi/CapsNet#discussion-groups)
- [Official Implementations](https://github.com/loretoparisi/CapsNet#official-implementations)
- [Proof of Work](https://github.com/loretoparisi/CapsNet#proof-of-work)
- [Implementations by Framework](https://github.com/loretoparisi/CapsNet#implementations-by-framework)
- [Implementations By Dataset](https://github.com/loretoparisi/CapsNet#implementations-by-dataset)
- [Implementations By Task](https://github.com/loretoparisi/CapsNet/blob/master/README.md#implementations-by-task)
- [Translations](https://github.com/loretoparisi/CapsNet/blob/master/README.md#translations)
- [Other Resources](https://github.com/loretoparisi/CapsNet/blob/master/README.md#other-resources)

# What's New
## Papers
- [CAPSULES WITH INVERTED DOT-PRODUCT ATTENTION ROUTING](https://openreview.net/pdf?id=HJe6uANtwH), Yao-Hung Hubert Tsai,
, Nitish Srivastava, Hanlin Goh, Ruslan Salakhutdinov, ICLR 2020 :new:

## Implementations By Dataset
### Toxic Comment Challenge (Kaggle)
- [chongjiujjin/capsule-net-with-gru](https://www.kaggle.com/chongjiujjin/capsule-net-with-gru) :fire:

# Abstract
We cover here the last and most interesting paper's abstract about Capsule Networks.

*We introduce a new routing algorithm for capsule networks, in which a child capsule is routed to a parent based only on agreement between the parent's state and the child's vote. The new mechanism 1) designs routing via inverted dot-product attention; 2) imposes Layer Normalization as normalization; and 3) replaces sequential iterative routing with concurrent iterative routing. When compared to previously proposed routing algorithms, our method improves performance on benchmark datasets such as CIFAR-10 and CIFAR-100, and it performs at-par with a powerful CNN (ResNet-18) with 4x fewer parameters. On a different task of recognizing digits from overlayed digit images, the proposed capsule model performs favorably against CNNs given the same number of layers and neurons per layer. We believe that our work raises the possibility of applying capsule networks to complex real-world tasks. Our code is publicly available at: https://github.com/apple/ml-capsules-inverted-attention-routing An alternative implementation is available at: https://github.com/yaohungt/Capsules-Inverted-Attention-Routing/blob/master/README.md*

Excerpt from
[CAPSULES WITH INVERTED DOT-PRODUCT ATTENTION ROUTING](https://openreview.net/pdf?id=HJe6uANtwH), Yao-Hung Hubert Tsai,
, Nitish Srivastava, Hanlin Goh, Ruslan Salakhutdinov, ICLR 2020 :new:

[UP](https://github.com/loretoparisi/CapsNet#CapsNet)

# Documentation
## Papers
- [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829v1), Sara Sabour, Nicholas Frosst, Geoffrey E Hinton, 26 Oct 2017
- [MATRIX CAPSULES WITH EM ROUTING](https://openreview.net/pdf?id=HJWLfGWRb), Anonymous authors, ICLR 2018
- [Does the Brain do Inverse Graphics?](http://helper.ipam.ucla.edu/publications/gss2012/gss2012_10754.pdf), Geoffrey Hinton, Alex Krizhevsky, Navdeep Jaitly, Tijmen Tieleman & Yichuan Tang, Department of Computer Science, University of Toronto, GSS 2012. 
- [CapsuleGAN: Generative Adversarial Capsule Network](https://arxiv.org/abs/1802.06167) Ayush Jaiswal, Wael AbdAlmageed, Premkumar Natarajan, USC Information Sciences Institute, Feb 2018
- [Brain Tumor Type Classification via Capsule Networks](https://arxiv.org/abs/1802.10200v2) Parnian Afshar, Arash Mohammadi, Konstantinos N. Plataniotis
- [Improved Explainability of Capsule Networks: Relevance Path by Agreement](https://arxiv.org/abs/1802.10204v1) Atefeh Shahroudnejad, Arash Mohammadi, Konstantinos N. Plataniotis
- [Investigating Capsule Networks with Dynamic Routing for Text Classification](https://arxiv.org/abs/1804.00538v2) Wei Zhao, Jianbo Ye, Min Yang, Zeyang Lei, Suofei Zhang, Zhou Zhao
- [
Capsules for Object Segmentation](https://arxiv.org/abs/1804.04241v1) Rodney LaLonde, Ulas Bagci
- [Investigating Capsule Networks with Dynamic Routing for
Text Classification](https://arxiv.org/pdf/1804.00538.pdf) Wei Zhao, Jianbo Ye3, Min Yang1, Zeyang Lei4 , Soufei Zhang5 , Zhou Zhao6
- [The Multi-Lane Capsule Network (MLCN)](https://www.researchgate.net/publication/331225145_The_Multi-Lane_Capsule_Network_MLCN) Vanderson M. do Rosario, Edson Borin, Mauricio Bretenitz Jr., Feb 2019
- [From Attention in Transformers to Dynamic Routing in Capsule Nets](https://staff.fnwi.uva.nl/s.abnar/?p=108)
- [TextCaps : Handwritten Character Recognition with Very Small Datasets](https://arxiv.org/pdf/1904.08095.pdf)
- [DeepCaps: Going Deeper with Capsule Networks](https://arxiv.org/abs/1904.09546)
- [Avoiding Implementation Pitfalls of "Matrix Capsules with EM Routing" by Hinton et al.](https://arxiv.org/abs/1907.00652) Ashley Gritzman, Aug 2019
- [A Neural-Symbolic Architecture for Inverse Graphics Improved by Lifelong Meta-Learning](https://arxiv.org/abs/1905.08910)
- [Adding Intuitive Physics to Neural-Symbolic Capsules Using Interaction Networks](https://arxiv.org/abs/1905.09891):new:
- [Capsule Routing via Variational Bayes](https://arxiv.org/abs/1905.11455)
- [Building Deep, Equivariant Capsule Networks](https://arxiv.org/abs/1908.01300)
- [Stacked Capsule Autoencoders](https://arxiv.org/abs/1906.06818), Adam R. Kosiorek, Sara Sabour, Yee Whye Teh, Geoffrey E. Hinton, 17 Jun 2019, revised 2 Dec 2019
- [Capsule Routing via Variational Bayes](https://arxiv.org/abs/1905.11455), Fabio De Sousa Ribeiro, Georgios Leontidis, Stefanos Kollias, 27 May 2019 (v1), last revised 3 Dec 2019
- [CAPSULES WITH INVERTED DOT-PRODUCT ATTENTION ROUTING](https://openreview.net/pdf?id=HJe6uANtwH), Yao-Hung Hubert Tsai,
, Nitish Srivastava, Hanlin Goh, Ruslan Salakhutdinov, ICLR 2020 :new:

[UP](https://github.com/loretoparisi/CapsNet#CapsNet)

## Articles
- [Understanding Hinton’s Capsule Networks. Part I: Intuition.](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b)
- [Understanding Hinton’s Capsule Networks. Part II: How Capsules Work.](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-ii-how-capsules-work-153b6ade9f66)
- [Understanding Hinton’s Capsule Networks. Part III: Dynamic Routing Between Capsules.](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-iii-dynamic-routing-between-capsules-349f6d30418)
- [Understanding Hinton’s Capsule Networks. Part IV: CapsNet Architecture](https://medium.com/@pechyonkin/part-iv-capsnet-architecture-6a64422f7dce) :new:
- [What is a CapsNet or Capsule Network?](https://hackernoon.com/what-is-a-capsnet-or-capsule-network-2bfbe48769cc)
- [Understanding Dynamic Routing between Capsules (Capsule Networks)](https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/)
- [Matrix capsules with EM routing](https://blog.acolyer.org/2017/11/14/matrix-capsules-with-em-routing/)
- [Capsule Networks Explained](https://kndrck.co/posts/capsule_networks_explained/)
- [Capsule Networks Are Shaking up AI — Here’s How to Use Them](https://hackernoon.com/capsule-networks-are-shaking-up-ai-heres-how-to-use-them-c233a0971952)
- [Understanding Capsule Networks — AI’s Alluring New Architecture](https://medium.freecodecamp.org/understanding-capsule-networks-ais-alluring-new-architecture-bdb228173ddc)

[UP](https://github.com/loretoparisi/CapsNet#CapsNet)

## Tutorials
- [Understand and apply CapsNet on Traffic sign classification](https://becominghuman.ai/understand-and-apply-capsnet-on-traffic-sign-classification-a592e2d4a4ea)
- [Running CapsuleNet on TensorFlow](https://medium.com/botsupply/running-capsulenet-on-tensorflow-1099f5c67189)

## Presentations
- [Dynamic Routing Between Capsules](https://www.slideshare.net/kyuhwanjung/vuno-dl-seminarcapsnetskyuhwanjung20171109)

## Webinar
- [Capsule Networks: An Improvement to Convolutional Networks - Siraj Raval
](https://www.youtube.com/watch?v=VKoLGnq15RM)

# Discussion Groups
- [CapsNet-Tensorflow](https://gitter.im/CapsNet-Tensorflow/Lobby)
- [Capsule Networks discussion - Facebook](https://www.facebook.com/groups/1830303997268623)
- [Could GANs work with Hinton's capsule theory?](https://www.quora.com/Could-GANs-work-with-Hintons-capsule-theory) :new:

# Official Implementations
The implementations has been considered to be official since the authors were directly involved in the papers as co-authors or they had some references.
- [Sarasra/models](https://github.com/Sarasra/models)
- [apple/ml-capsules-inverted-attention-routing](https://github.com/apple/ml-capsules-inverted-attention-routing)

# Proof of Work
- [Adversarial Attack to Capsule Networks](https://github.com/jaesik817/adv_attack_capsnet) :new:

# Other Resources
- [A curated list of awesome resources related to capsule networks](https://github.com/aisummary/awesome-capsule-networks)

# Implementations by Framework
## Pytorch
- [mavanb/capsule_network_pytorch](https://github.com/mavanb/capsule_network_pytorch)
- [leftthomas/CapsNet](https://github.com/leftthomas/CapsNet)
- [dragen1860/CapsNet-Pytorch](https://github.com/dragen1860/CapsNet-Pytorch)
- [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks)
- [andreaazzini/capsnet.pytorch](https://github.com/andreaazzini/capsnet.pytorch)
- [CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch)
- [Ujjwal-9/CapsNet](https://github.com/Ujjwal-9/CapsNet)
- [cedrickchee/capsule-net-pytorch](https://github.com/cedrickchee/capsule-net-pytorch)
- [laubonghaudoi/CapsNet_guide_PyTorch](https://github.com/laubonghaudoi/CapsNet_guide_PyTorch)
- [tonysy/CapsuleNet-PyTorch](https://github.com/tonysy/CapsuleNet-PyTorch)
- [AlexHex7/CapsNet_pytorch](https://github.com/AlexHex7/CapsNet_pytorch)
- [leftthomas/CCN](https://github.com/leftthomas/CCN)
- [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks)
- [fabio-deep/Variational-Capsule-Routing](https://github.com/fabio-deep/Variational-Capsule-Routing)
- [apple/ml-capsules-inverted-attention-routing](https://github.com/apple/ml-capsules-inverted-attention-routing) :new:

## Pytorch + CUDA
- [leftthomas/CapsuleLayer](https://github.com/leftthomas/CapsuleLayer)
- [bakirillov/capsules](https://github.com/bakirillov/capsules)

## Jupyter Notebook
- [acburigo/CapsNet](https://github.com/acburigo/CapsNet)
- [leoniloris/CapsNet](https://github.com/leoniloris/CapsNet)
- [ruslangrimov/capsnet-with-capsulewise-convolution](https://github.com/ruslangrimov/capsnet-with-capsulewise-convolution)
- [aliasvishnu/Capsule-Networks-Notebook-MNIST](https://github.com/aliasvishnu/Capsule-Networks-Notebook-MNIST)
- [adambielski/CapsNet-pytorch](https://github.com/adambielski/CapsNet-pytorch)
- [rrqq/CapsNet-tensorflow-jupyter](https://github.com/rrqq/CapsNet-tensorflow-jupyter)

## Torch
- [leftthomas/FCCapsNet](https://github.com/leftthomas/FCCapsNet)

## Tensorflow
- [IBM/matrix-capsules-with-em-routing](https://github.com/IBM/matrix-capsules-with-em-routing)
- [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)
- [InnerPeace-Wu/CapsNet-tensorflow](https://github.com/InnerPeace-Wu/CapsNet-tensorflow)
- [laodar/tf_CapsNet](https://github.com/laodar/tf_CapsNet)
- [nnUyi/CapsNet](https://github.com/nnUyi/CapsNet)
- [tjiang31/CapsNet](https://github.com/tjiang31/CapsNet)
- [rrqq/CapsNet-tensorflow-jupyter](https://github.com/rrqq/CapsNet-tensorflow-jupyter)
- [winwinJJiang/capsNet-Tensorflow](https://github.com/winwinJJiang/capsNet-Tensorflow)
- [etendue/CapsNet_TF](https://github.com/etendue/CapsNet_TF)
- [thibo73800/capsnet_traffic_sign_classifier](https://github.com/thibo73800/capsnet_traffic_sign_classifier)
- [bourdakos1/capsule-networks](https://github.com/bourdakos1/capsule-networks)
- [jostosh/capsnet](https://github.com/jostosh/capsnet)
- [alisure-ml/CapsNet](https://github.com/alisure-ml/CapsNet)
- [bourdakos1/CapsNet-Visualization](https://github.com/bourdakos1/CapsNet-Visualization) 
- [andyweizhao/capsule_text_classification](https://github.com/andyweizhao/capsule_text_classification) :new:

## Keras
- [XifengGuo/CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras)
- [XifengGuo/CapsNet-Fashion-MNIST](https://github.com/XifengGuo/CapsNet-Fashion-MNIST)
- [sunxirui310/CapsNet-Keras](https://github.com/sunxirui310/CapsNet-Keras)
- [theblackcat102/dynamic-routing-capsule-cifar](https://github.com/theblackcat102/dynamic-routing-capsule-cifar)
- [streamride/CapsNet-keras-imdb](https://github.com/streamride/CapsNet-keras-imdb)
- [fengwang/minimal-capsule](https://github.com/fengwang/minimal-capsule)
- [vinojjayasundara/textcaps](https://github.com/vinojjayasundara/textcaps)
- [Kayzaks/VividNet](https://github.com/Kayzaks/VividNet) :new:

[UP](https://github.com/loretoparisi/CapsNet#CapsNet)

## MXNet
- [AaronLeong/CapsNet_Mxnet)](https://github.com/AaronLeong/CapsNet_Mxnet)
- [Soonhwan-Kwon/capsnet.mxnet](https://github.com/Soonhwan-Kwon/capsnet.mxnet)
- [GarrickLin/Capsnet.Gluon](https://github.com/GarrickLin/Capsnet.Gluon)

## CNTK
- [Southworkscom/CapsNet-CNTK](https://github.com/southworkscom/CapsNet-CNTK)

## Lasagne
- [DeniskaMazur/CapsNet-Lasagne](https://github.com/DeniskaMazur/CapsNet-Lasagne)

## Chainer
- [soskek/dynamic_routing_between_capsules](https://github.com/soskek/dynamic_routing_between_capsules)

## Matlab
- [yechengxi/LightCapsNet](https://github.com/yechengxi/LightCapsNet)

## R
- [dfalbel/capsnet](https://github.com/dfalbel/capsnet)

## C++
- [MxAR/CapsNet.cpp](https://github.com/MxAR/CapsNet.cpp)

## C
- [MxAR/CapsNet.c](https://github.com/MxAR/CapsNet.c)

## JavaScript
- [alseambusher/capsnet.js](https://github.com/alseambusher/capsnet.js)

## Vulcan
- [moothyknight/CapsNet-for-Graphics-Rendering-Optimization](https://github.com/moothyknight/CapsNet-for-Graphics-Rendering-Optimization)

## Other
- [jaesik817/adv_attack_capsnet](https://github.com/jaesik817/adv_attack_capsnet)

[UP](https://github.com/loretoparisi/CapsNet#CapsNet)

# Implementations By Dataset
## MNIST
- [leftthomas/CapsNet](https://github.com/leftthomas/CapsNet)
- [dragen1860/CapsNet-Pytorch](https://github.com/dragen1860/CapsNet-Pytorch)
- [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks)
- [andreaazzini/capsnet.pytorch](https://github.com/andreaazzini/capsnet.pytorch)
- [CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch)
- [Ujjwal-9/CapsNet](https://github.com/Ujjwal-9/CapsNet)
- [cedrickchee/capsule-net-pytorch](https://github.com/cedrickchee/capsule-net-pytorch)
- [laubonghaudoi/CapsNet_guide_PyTorch](https://github.com/laubonghaudoi/CapsNet_guide_PyTorch)
- [tonysy/CapsuleNet-PyTorch](https://github.com/tonysy/CapsuleNet-PyTorch)
- [leftthomas/FCCapsNet](https://github.com/leftthomas/FCCapsNet)
- [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)
- [InnerPeace-Wu/CapsNet-tensorflow](https://github.com/InnerPeace-Wu/CapsNet-tensorflow)
- [laodar/tf_CapsNet](https://github.com/laodar/tf_CapsNet)
- [nnUyi/CapsNet](https://github.com/nnUyi/CapsNet)
- [tjiang31/CapsNet](https://github.com/tjiang31/CapsNet)
- [rrqq/CapsNet-tensorflow-jupyter](https://github.com/rrqq/CapsNet-tensorflow-jupyter)
- [winwinJJiang/capsNet-Tensorflow](https://github.com/winwinJJiang/capsNet-Tensorflow)
- [etendue/CapsNet_TF](https://github.com/etendue/CapsNet_TF)
- [Southworkscom/CapsNet-CNTK](https://github.com/southworkscom/CapsNet-CNTK)

## IMDB Reviews
- [streamride/CapsNet-keras-imdb](https://github.com/streamride/CapsNet-keras-imdb)

## Cifar 10
- [theblackcat102/dynamic-routing-capsule-cifar](https://github.com/theblackcat102/dynamic-routing-capsule-cifar)

## BanglaLekha-Isolated Dataset:
- [codeheadshopon/CapsNet_BanglaLekha](https://github.com/codeheadshopon/CapsNet_BanglaLekha)

## Traffic Sign Dataset (German):
- [thibo73800/capsnet_traffic_sign_classifier](https://github.com/thibo73800/capsnet_traffic_sign_classifier)

## Iceberg Classification Challenge (Kaggle)
- [sdhayalk/CapsNet-for-Iceberg-or-Submarine-Classification](https://github.com/sdhayalk/CapsNet-for-Iceberg-or-Submarine-Classification)
- [leftthomas/IcebergClassifier](https://github.com/leftthomas/IcebergClassifier)

## Toxic Comment Challenge (Kaggle)
- [chongjiujjin/capsule-net-with-gru](https://www.kaggle.com/chongjiujjin/capsule-net-with-gru)

[UP](https://github.com/loretoparisi/CapsNet#CapsNet)

# Implementations by Task
## Text Classification
- [andyweizhao/capsule_text_classification](https://github.com/andyweizhao/capsule_text_classification) :new:
## Speech Recognition
- [SvenDH/CapsNet-ASR](https://github.com/SvenDH/CapsNet-ASR)
## Emotion Recognition
- [mitiku1/Emopy-CapsNet](https://github.com/mitiku1/Emopy-CapsNet)
## Named Entity Recognition (NER)
- [Chucooleg/CapsNet_for_NER](https://github.com/Chucooleg/CapsNet_for_NER)
## Natural Language Processing (NLP)
- [stefan-it/capsnet-nlp](https://github.com/stefan-it/capsnet-nlp)

[UP](https://github.com/loretoparisi/CapsNet#CapsNet)

# Translations
## Japanese
- [CapsNetについての調べ - (JAPANESE)](https://qiita.com/onlyzs/items/5096f50a21758a536d9a)

## Turkish
- [deeplearningturkiye/kapsul-agi-capsule-network](https://github.com/deeplearningturkiye/kapsul-agi-capsule-network)

[UP](https://github.com/loretoparisi/CapsNet#CapsNet)
