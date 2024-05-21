# GradTrust
[MIPR 2024 Invited Paper] Code for the paper: Counterfactual Gradients-based Quantification of Prediction Trust in Neural Networks.

Work conducted at [OLIVES@GaTech](https://alregib.ece.gatech.edu). Arxiv paper available at (Will update this once it is published on Arxiv)

Official code repository for the paper: M. Prabhushankar and G. AlRegib, "Counterfactual Gradients-based Quantification of Prediction Trust in Neural Networks", In 2024 IEEE 7th International Conference on Multimedia Information Processing and Retrieval (MIPR), San Jose, CA, Aug. 7-9, 2024 (Invited Paper).

![Concept image showcasing value of GradTrust over Softmax](figs/Concept.png)
Scatter plot between the proposed GradTrust on x-axis and softmax confidence on y-axis on ImageNet validation dataset using ResNet-18. Green points indicate correctly classified data and red indicates misclassified data. Representative misclassified and correctly images in the numbered boxes are displayed alongside the scatterplot, with their predictions (in red) and labels (in blue).

## Abstract
The widespread adoption of deep neural networks in machine learning calls for an objective quantification of esoteric trust. In this paper we propose **GradTrust**, a classification trust measure for large-scale neural networks at inference. The proposed method utilizes variance of counterfactual gradients, i.e. the required changes in the network parameters if the label were different. We show that GradTrust is superior to existing techniques for detecting misprediction rates on 50000 images from ImageNet validation dataset. Depending on the network, GradTrust detects images where either the ground truth is incorrect or ambiguous, or the classes are co-occurring. We extend GradTrust to Video Action Recognition on Kinetics-400 dataset. We showcase results on 14 architectures pretrained on ImageNet and 5 architectures pretrained on Kinetics-400. We observe the following: (i) simple methodologies like negative log likelihood and margin classifiers outperform state-of-the-art uncertainty and out-of-distribution detection techniques for misprediction rates, and (ii) the proposed GradTrust is in the Top-2 performing methods on 37 of the considered 38 experimental modalities.

## Usage
The repository provides a demo code that quantifies prediction trust using GradTrust as well as 8 comparison metrics for any given pretrained model.

### Getting Started
Clone the repository and run the following commands to create a conda envirnoment and install all dependencies.
```
conda create -n gradtrust python=3.6
conda activate gradtrust
cd GradTrust
conda install pytorch torchvision -c pytorch
pip install -r requirements.in
```

### Evaluation
The code requires the user to feed in the network for Trust Quantification. It is specifically written for pretrained models available at [PyTorch's Torchvision Library](https://pytorch.org/vision/stable/models.html). Feed in the name of the network as defined in the above link. For instance, if we the user wants trust quantification for ResNet-18, run the following:

```
python demo.py --network 'resnet18'
```

## Results on Demo code

For the default network > vit_b_16, the following results will be printed for the water-bird.JPEG image:
```
The prediction is : [129] with GradTrust: 994.78125
Comparison Metrics:
Softmax Confidence: [0.904579]
Entropy: [-0.9696515]
Margin: [0.9039816]
Log-likelihood: 9.182098
ODIN: 0.001096937
MC-Dropout: [-0.90219054]
Purview (Initial layers): -0.0009888242
Purview (Final layers): -0.0025726024
Grad Norm: -0.009952871
```
## Results when applied to Full ImageNet Validation set
### Quantitative Results
![Quantitative results showcasing value of GradTrust across 14 networks](figs/Quantitative.png)
Results on 50000 images from ImageNet 2012 validation dataset. AUAC and AUFC values are shown for each metric. The top-2 AUAC and AUFC values for every network are bolded. Rows are ordered based on increasing overall accuracy.

### Qualitative Results
![Qualitative results showcasing value of GradTrust on two networks](figs/Qualitative.png)
Qualitative analysis of mispredictions on AlexNet (top row), MaxVit-t (middle row) and ensemble mispredictions across all networks (bottom row). All displayed images have high softmax and ordered in ascending order of GradTrust.

## Questions?

If you have any questions, regarding the dataset or the code, you can contact the authors [(mohit.p@gatech.edu)](mohit.p@gatech.edu), or even better open an issue in this repo and we'll do our best to help.
