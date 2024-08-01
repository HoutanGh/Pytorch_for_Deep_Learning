# Pytorch_for_Deep_Learning

# Model Files:

## first_model.py
- Introduces a basic linear regression model using PyTorch.
- Demonstrates creating, training, and evaluating the model.
- Includes functionality for saving and loading the model.

## first_model_with_agnostic_code.py
- Similar to "first_model.py" but adapted for agnostic device usage (CPU/GPU).
- Includes training, evaluation, and model persistence.
- Visualizes predictions and checks for model consistency after loading.
  
## binary_circle_model.py
- Implements a binary classification model using circular data generated with `make_circles`.
- Covers data preparation, model definition, training, and evaluation.
- Introduces non-linearity using ReLU activation functions and evaluates model performance.

## multiclass_blob_model.py
- Implements a multiclass classification model using blob data generated with `make_blobs`.
- Covers data preparation, model definition, training, and evaluation.
- Uses softmax for converting logits to probabilities and measures accuracy.

## clothes_CNN_model.py
- Implements a Convolutional Neural Network (CNN) for FashionMNIST data.
- Details model architecture with convolutional layers and pooling.
- Uses helper functions for training, testing, and evaluation.
- Incorporates functions for saving and loading the model.
  
## clothes_agnostic_model.py
- Implements a simple neural network model for classifying FashionMNIST data, just to show how this method does not really work.
- Main aim is to demonstrate how to adapt code for agnostic device usage (CPU/GPU).
- Includes training and testing functions and measures training time.

## food_model_simple.py
- Implements a CNN for classifying images of food (pizza, steak, sushi).
- Demonstrates data preparation, transformation, and loading with custom datasets.
- Defines a `TinyVGG` model for image classification.
- Includes training, evaluation, and prediction functions.
- Utilizes helper functions for visualizing training progress and making predictions on new images.



## my_helper_functions.py
- Provides helper functions for training, testing, and evaluating PyTorch models.
- Includes functions for calculating accuracy, performing training and testing steps, and evaluating models.
- Used across multiple models for consistent and reusable code.

# Note files:
## chapter_0_notes.py: Basic Tensor Operations
- Introduction to fundamental tensor operations using PyTorch.
- Covers creating tensors, reshaping, stacking, and indexing.
- Demonstrates moving tensors between CPU and GPU.

## chapter_1_notes.py: Linear Regression
- Focus on implementing a simple linear regression model with PyTorch.
- Includes creating and splitting a dataset into training and test sets.
- Defines the linear regression process and parameters.

## chapter_2_notes.py: Neural Network Classification
- Covers binary, multiclass, and multilabel classification tasks.
- Explains neural network model implementation for classification.
- Includes data preparation, training loops, and evaluation methods.
- Introduces non-linearity to improve model performance.

## chapter_3_notes.py: Convolutional Neural Networks (CNN)
- Implementation of Convolutional Neural Networks for image classification.
- Uses the FashionMNIST dataset for practical demonstrations.
- Details data preparation, model definition, training, and evaluation.
- Advanced techniques: saving/loading models and confusion matrices for performance evaluation.

## chapter_4_notes.py: Custom Datasets and Data Augmentation
- Focus on creating and handling custom datasets (pizza, steak, sushi images).
- Demonstrates downloading, extracting, and visualizing a custom dataset.
- Explains data transformations and creating a custom dataset class.
- Implements and trains a CNN model on the custom dataset.
- Discusses data augmentation techniques to enhance model robustness and performance.
