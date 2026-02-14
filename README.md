# MNIST Handwritten Digit Classifier (PyTorch)

This project implements a handwritten digit classifier using **PyTorch** and **TorchVision** on the **MNIST dataset**.
The goal of the project is to build a complete deep-learning training pipeline, monitor training behavior using **TensorBoard**, and address **overfitting using regularization**.

---

## Project Overview

The model is a simple **feed-forward neural network (ANN)** trained to classify grayscale handwritten digits (0–9).

This project demonstrates:

* Building neural networks in PyTorch
* Writing training and evaluation loops
* Using TensorBoard for experiment tracking
* Detecting and fixing overfitting
* Applying L2 regularization (weight decay)

---

## Model Architecture

The network consists of three fully connected layers:

Input (28×28 image → flattened to 784)
→ Linear layer
→ ReLU
→ Linear layer
→ ReLU
→ Linear layer (10 classes)

The final layer outputs **logits**, and training uses:

* **CrossEntropyLoss** (includes Softmax internally)
* **Adam optimizer**
* **L2 regularization (weight decay)**

---

## Training Setup

Dataset:

* MNIST (via TorchVision)

Optimizer:

* Adam

Loss function:

* CrossEntropyLoss

Regularization:

* L2 regularization (weight decay)

Experiment tracking:

* TensorBoard

---

## Overfitting Observation

During initial training runs, TensorBoard showed:

* Training loss decreasing continuously
* Test loss increasing near the end of training

This indicated **overfitting**.

The issue was resolved by adding **L2 regularization (weight decay)** to the optimizer, which improved generalization and stabilized validation performance.

---

## Results

Final performance:

* Training accuracy: **98%**
* Test accuracy: **97.8%**

TensorBoard was used to visualize:

* Training vs test loss curves
* Accuracy curves
* Model predictions vs ground-truth labels

---

## Running the Project

Install dependencies:

uv add torch torchvision tensorboard matplotlib

Train the model:

python train.py

Launch TensorBoard:

tensorboard --logdir runs

Then open:
http://localhost:6006

---

## Project Structure

MNIST_Pytorch/
│
├── MNIST.py
├── runs/
└── README.md


Built as part of a deep learning learning journey using PyTorch.
