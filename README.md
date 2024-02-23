# Description of model.py

## Overview
The `model.py` file contains code for defining a neural network model for the MNIST dataset using PyTorch.

## Contents
1. **Data Loading and Preprocessing**
    - The file imports necessary libraries from PyTorch for data loading and preprocessing.
    - It defines transformations for both training and testing data.
    - MNIST dataset is loaded using `datasets.MNIST` and transformed accordingly for training and testing sets.
    - DataLoaders for both training and testing sets are created.

2. **Neural Network Model Definition**
    - A custom neural network model `Net` is defined inheriting from `nn.Module`.
    - The neural network consists of several convolutional layers (`Conv2d`) and fully connected layers (`Linear`).
    - The architecture comprises four convolutional layers followed by two fully connected layers.
    - ReLU activation functions are used after each convolutional layer.
    - Log Softmax activation is applied at the output layer.

3. **Visualization**
    - The code includes a visualization of a batch of training data using Matplotlib.


# Description of utils.py

## Overview
The `utils.py` file contains utility functions and code for training and testing a neural network model on the MNIST dataset using PyTorch.

## Contents
1. **Data Loading and Preprocessing**
    - Similar to `model.py`, this file imports necessary libraries from PyTorch for data loading and preprocessing.
    - It defines transformations for both training and testing data.
    - MNIST dataset is loaded using `datasets.MNIST` and transformed accordingly for training and testing sets.
    - The code also initializes lists to store training and testing losses, accuracies, and incorrectly predicted samples.

2. **Utility Functions**
    - `GetCorrectPredCount`: This function calculates the number of correct predictions given the predicted values and ground truth labels.

3. **Training and Testing Functions**
    - `train`: This function performs the training of the neural network model.
        - It iterates over batches of data from the training loader.
        - Calculates the loss, performs backpropagation, and updates model parameters.
        - Updates training loss and accuracy metrics.
        
    - `test`: This function evaluates the performance of the model on the test dataset.
        - It iterates over batches of data from the test loader.
        - Computes test loss and accuracy metrics.

4. **Training Loop**
    - The file includes a training loop where the model is trained and tested for a specified number of epochs.
    - It initializes the model, optimizer, scheduler, loss function, and other necessary components.
    - Training and testing are performed for each epoch, and the learning rate scheduler is updated.
    - Finally, it plots the training and testing losses as well as accuracies for visualization.

