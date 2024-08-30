# Brain Tumor Classification using CNN and Transfer Learning
# Introduction
This project focuses on classifying brain tumors from MRI images using deep learning techniques. The goal is to build a model that can accurately categorize images into different types of brain tumors, including glioma, meningioma, and pituitary tumors, or identify images as notumor.

# Objective:
To develop and evaluate convolutional neural network (CNN) models and transfer learning models for the classification of brain tumor images.

# Dataset:
The dataset consists of MRI images categorized into four classes: glioma, meningioma, pituitary, and notumor. The images are divided into training, validation, and test sets.

# Installation Instructions:
Prerequisites:
Python 3.6 or higher
TensorFlow
Keras
Matplotlib
Seaborn
Pandas
Visualkeras


# Data Preparation:
Data Loading: The images are loaded from the Google Drive directories and organized into DataFrames with their respective class labels. The generate_dataframe_from_directory function is used to load the images.

Data Augmentation:
Data augmentation techniques such as rescaling, brightness adjustments, and validation/test preprocessing are applied using TensorFlow's ImageDataGenerator.

# Model Architecture:
# CNN Model:
A custom CNN model is built with multiple convolutional layers, batch normalization, max-pooling, dropout, and fully connected layers. The model is designed to handle the image input size of 299x299 pixels.

![download (96)](https://github.com/user-attachments/assets/1efe8685-64c2-4432-b62a-b9760bd91838)

# Transfer Learning Models:
Pre-trained models such as InceptionV3 and VGG19 are used for transfer learning. The top layers of these models are replaced with custom fully connected layers suitable for the classification task.

InceptionV3 Model
![download (97)](https://github.com/user-attachments/assets/2fcec961-1a53-4160-b3f3-2e772c9596b8)


VGG19 Model:
![download (98)](https://github.com/user-attachments/assets/670598a3-07d3-48ab-9c16-0eb9ff9367a4)


# Training:
Training Process: The models are trained using categorical cross-entropy loss and Adamax optimizer. Early stopping and learning rate scheduling are employed to optimize the training process.

Training History: The training and validation accuracy and loss are plotted over the epochs to monitor the performance. Below is the examploe of CNN Model Training.
![download (99)](https://github.com/user-attachments/assets/40419398-7ae9-408b-aad2-49cd6131ceb6)

# Model Evaluation on the Test Set:

![image](https://github.com/user-attachments/assets/071392cd-da53-4967-ab53-66be29b084e9)

# Conclusion 
In this experiment, we investigated the efficacy of several deep learning models for classifying brain cancers from MRI data. The bespoke CNN model had the greatest accuracy on the test set, 98.63%, indicating its outstanding capacity to discriminate between different forms of brain tumors. The transfer learning models, InceptionV3 and VGG19, also scored well, with test accuracies of 96.34% and 95.27%, respectively. Despite having lesser accuracy than a bespoke CNN, these models gave a major benefit by using pre-trained knowledge, reducing training time and computer resources required. The findings emphasize the promise of deep learning approaches in medical image processing, providing significant tools for aiding in the identification and treatment of brain cancers.Future research might concentrate on further refining these models, investigating more sophisticated architectures, or applying these models to bigger and more diverse datasets to increase generalizability and resilience.




