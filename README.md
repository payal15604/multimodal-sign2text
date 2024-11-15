## Benchmarking Multi-Model Architecture Comparison for High-Accuracy Sign Language to Text Mapping


## Abstract

This project explores the potential of various machine learning models to accurately translate American Sign Language into written text, aiming to enhance communication between the hearing and the deaf-mute communities. Current systems often lack adaptability and scalability, facing challenges with dataset specificity and limited flexibility for new signs or languages. Through a comparative analysis of model architectures, we evaluate performance and generalizability to identify the most effective approaches for real-world sign language translation. Our work contributes to creating robust sign-to-text models that foster inclusive communication solutions.

## Introduction

Sign language is a critical communication tool for people with hearing or speech impairments. Automated Sign Language Recognition (SLR) systems can support real-time translation and assistive devices, bridging communication gaps between sign language users and the broader community. This project examines various deep learning architectures—Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and others—to improve accuracy and flexibility in SLR.

## Methodology

The study evaluates several machine learning models, focusing on CNNs for capturing spatial features from ASL gestures. Key techniques include:

- **CNN Architecture**: Two convolutional blocks, each followed by max-pooling and dropout layers, capture image features.
- **Data Augmentation**: Random rotations, zooms, and shifts simulate real-world variability in gestures, enhancing model robustness.
- **Training & Optimization**: The model is trained with the RMSprop optimizer, using categorical cross-entropy loss and dynamic learning rate adjustments to improve accuracy and prevent overfitting.

The model is evaluated using a standardized ASL dataset and tested for generalization on new data to ensure scalability.

## Implementation

### Dataset

The **American Sign Language** dataset from Kaggle, consisting of grayscale images scaled to 28x28 pixels, represents letters A-I and K-Y (excluding J and Z due to motion). The dataset provides labeled images for training machine learning models to recognize static hand gestures.

### Preprocessing

1. **Image Resizing**: All images are resized to 28x28 pixels for uniform input dimensions.
2. **Grayscale Conversion**: Images are converted to grayscale to reduce complexity, focusing on shape rather than color.
3. **Normalization**: Pixel values are normalized from [0, 255] to [0, 1] to enhance model convergence.

### Model Architecture

The CNN model consists of:

- **Convolutional Layers**: Two convolutional layers with 128 filters each, using a \(5 \times 5\) kernel size and ReLU activation.
- **Pooling and Dropout**: MaxPooling layers for dimensionality reduction, and Dropout layers for regularization.
- **Dense Layers**: Fully connected layers for classification with softmax activation.

The model is compiled with RMSprop optimizer and categorical cross-entropy loss, training for 25 epochs with dynamic learning rate adjustment.

## Algorithm

The algorithm for sign language recognition includes:

1. **Data Preprocessing**: Reshaping and normalizing images.
2. **Model Definition**: Setting up a sequential CNN with Conv2D, MaxPooling, and Dropout layers.
3. **Training**: Optimizing with RMSprop and categorical cross-entropy loss, training with data augmentation.
4. **Evaluation**: Testing model accuracy and generating predictions on test data.


## Future Work

This project aims to improve adaptability across diverse sign languages and generalize beyond the ASL dataset. Future work includes:

- Incorporating additional sign languages
- Exploring transfer learning to adapt models with minimal retraining
- Integrating dynamic gestures for a comprehensive SLR solution


## Contributing

We are grateful to everyone who contributed to this project:

- **Payal** - Student Researcher 
- **Anureet Kaur** - Student Researcher
- **Dr. Jyoti Maggu** - Project Guidance and Oversight



## Setting up the Environment Variables

To run this project, you will need to add the following environment variables.

### Clone the repository:

```bash
git clone <https://github.com/payal15604/multimodal-sign2text>
cd <multimodal-sign2text>
```

### Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install the required packages:
```bash
pip install -r requirements.txt
```