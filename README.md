# GradTrust
[MIPR 2024 Invited] Code for the paper: Counterfactual Gradients-based Quantification of Prediction Trust in Neural Networks

![Concept image showcasing value of GradTrust over Softmax](figs/Concept.png)

## Abstract
he widespread adoption of deep neural networks in machine learning calls for an objective quantification of esoteric trust. In this paper we propose **GradTrust**, a classification trust measure for large-scale neural networks at inference. The proposed method utilizes variance of counterfactual gradients, i.e. the required changes in the network parameters if the label were different. We show that GradTrust is superior to existing techniques for detecting misprediction rates on 50000 images from ImageNet validation dataset. Depending on the network, GradTrust detects images where either the ground truth is incorrect or ambiguous, or the classes are co-occurring. We extend GradTrust to Video Action Recognition on Kinetics-400 dataset. We showcase results on 14 architectures pretrained on ImageNet and 5 architectures pretrained on Kinetics-400. We observe the following: (i) simple methodologies like negative log likelihood and margin classifiers outperform state-of-the-art uncertainty and out-of-distribution detection techniques for misprediction rates, and (ii) the proposed GradTrust is in the Top-2 performing methods on 37 of the considered 38 experimental modalities.

## Results

![Qualitative results showcasing value of GradTrust on three networks](figs/Qualitative.png)
