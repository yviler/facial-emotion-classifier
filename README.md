# Facial Emotion Classifier with EfficientNetV2B0 

This project is a deep learning-based facial expression classifier trained on grayscale emotion-labeled face images using a transfer learning pipeline built with TensorFlow and EfficientNetV2B0.

It leverages modern best practices in data loading, augmentation, regularization, and transfer learning to classify facial emotions into 7 categories.

## Dataset

The dataset consists of facial images categorized into 7 emotions:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

Images are grayscale but converted to RGB to be compatible with pretrained CNNs. The data is split using `image_dataset_from_directory` with a stratified 80/20 training/validation split.

## Model Architecture

- **Base Model**: [EfficientNetV2B0](https://arxiv.org/abs/2104.00298) pretrained on ImageNet
- **Input Shape**: `(224, 224, 3)`
- **Augmentations**: Horizontal flip, rotation, zoom, contrast, translation
- **Classifier Head**:  
  - Global Average Pooling  
  - Dropout  
  - Dense layer (7 units, softmax)

### Training Details

- Optimizer: Adam (`lr=1e-4`)
- Loss: `sparse_categorical_crossentropy`
- Callbacks:
  - EarlyStopping
  - ReduceLROnPlateau
  - ModelCheckpoint
- Batch Size: 32
- Epochs: 10 (initial)

## Performance (so far)

- **Training Accuracy**: ~91%
- **Validation Accuracy**: ~67%  
> Overfitting observed → currently testing stronger augmentation, dropout, and fine-tuning strategies.

This project demonstrates how to use deep learning tools effectively while being aware of real challenges like overfitting and model generalization.


## Requirements

```bash
tensorflow>=2.15
matplotlib
pillow

