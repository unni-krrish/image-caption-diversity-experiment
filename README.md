# Experiment : Caption Diversity and Image Captioning Model Performance
## Description
Captioning and generating descriptions about images has become an important problem in Artificial Intelligence as it combines multiple modes of learning from images and text. In this project, we aim to build a multimodal neural network for image captioning that can achieve reasonable accuracy while maintaining a simple architecture. While many of the previous works have focused on using only one caption per image in the training dataset, we intend to diversify our model training by using more captions per image while using the same set of images and test whether training diversity improves model performance.
## Quick Instructions
1) Download the entire COCO dataset, annotations from official site to local (Use VM)
2) Generate transfer values using EfficientNet-Bx models using 1 (save_train_data.py)
3) Prepare captions using keras tokenizer (save_train_data.py)
4) Get transfer values of validation dataset (save_validation_data.py)
5) Train the model using 2,3 (train_model.py)
6) Make predictions on validation images (predict.py)
7) Evaluate predictions using standard caption evaluation metrics (BLEU-4, METEOR, SPICE, and CIDEr)

### Architecture
![image](https://user-images.githubusercontent.com/53073761/131271792-97c00683-6a3f-47b4-ad52-23b0290c1a38.png)



### Results on [MSCOCO-2017](https://cocodataset.org/#download) dataset
![image](https://user-images.githubusercontent.com/53073761/131271732-0bfdd48b-e8e5-4f0a-b1f2-77d1629fd690.png)



## Requirements
* Tensorflow, [pycocoevalcaps](https://github.com/salaniz/pycocoevalcap) libraries
* As the storage and compute demands are fairly high for a local machine, use a Virtual Machine on [Azure](https://azure.microsoft.com/en-us/services/virtual-machines/) or [GCP](https://cloud.google.com/compute)
* Repeat the experiments by changing number of captions per image in the preprocessing stage. 

## Improvements Suggested
* Use of better object detection models (eg. EfficientNet-B6)
* Datasets with more than 5 captions per image to see the diminishing effect of caption diversity

## References
* [Jia, X. et al. 2015](https://arxiv.org/abs/1509.04942) Guiding Long-Short Term Memory for Image Caption Generation.
* [Wang, Y. et al. 2017](https://ieeexplore.ieee.org/document/8100263) Skeleton Key: Image Captioning by Skeleton-Attribute Decomposition
* [Wu, Q. et al. 2018](https://ieeexplore.ieee.org/abstract/document/7934440) Image Captioning and Visual Question Answering Based on Attributes and External Knowledge
