# Cross-layer Contrastive Learning of Latent Semantics for Facial Expression Recognition

Convolutional neural networks (CNNs) have achieved significant improvement for the task of facial expression recognition. However, current training still suffers from the inconsistent learning intensities among different layers, i.e., the feature representations in the shallow layers are not sufficiently learned compared with those in deep layers. To this end, this work proposes a contrastive learning framework to align the feature semantics of shallow and deep layers, followed by an attention module for representing the multi-scale features in the weight-adaptive manner. The proposed algorithm has three main merits. First, the learning intensity, defined as the magnitude of the backpropagation gradient, of the features on the shallow layer is enhanced by cross-layer contrastive learning. Second, the latent semantics in the shallow-layer and deep-layer features are explored and aligned in the contrastive learning, and thus the fine-grained characteristics of expressions can be taken into account for the feature representation learning. Third, by integrating the multi-scale features from multiple layers with an attention module, our algorithm achieved the state-of-the-art performances, i.e. 92.21\%, 89.50\%, 62.82\%, on three in-the-wild expression databases, i.e. RAF-DB, FERPlus, SFEW, and the second best performance, i.e. 65.29\% on AffectNet dataset.

![img](/architect.png)

# Train
Pytorch

Torch 1.1.0 or higher and Python 3.5 or higher are required.

python train_fs.py


If you find our code or paper useful, please cite as

@inproceedings{CCLLS,
  title={Cross-layer Contrastive Learning of Latent Semantics for Facial Expression Recognition},
}
