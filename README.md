# NumPy Image Recognition (MNIST)

A modular, from-scratch implementation of a neural network using **only NumPy**.  
No PyTorch, no TensorFlow — just mathematics, linear algebra, and code.



## 🚀 Why This Project?

This project was built to deeply understand how neural networks work internally, without relying on high-level machine learning frameworks.

Key goals:
- **From Scratch** – Implemented entirely using low-level math and NumPy
- **Modular Design** – Clean separation of model, activations, losses, and training logic
- **Manual Backpropagation** – Explicit chain rule and gradient computation
- **Mini-Batch SGD** – Custom batch generator for stable convergence
- **Visualization** – Automated plotting of loss and accuracy curves
- **He Initialization** – Proper weight initialization for deep network stability



## 🎯 When to Use This Project

- You want to understand the mathematics behind deep learning
- You need a lightweight MLP without heavy dependencies
- You want to see how backpropagation and gradient descent work under the hood
- You are building a portfolio to demonstrate core AI engineering skills



## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/Image-Recognition.git
cd Image-Recognition
pip install -r requirements.txt

## 2. Setup Dataset

This project uses the MNIST dataset in CSV format.

- Download `train.csv` from Kaggle (Digit Recognizer)
- Place `train.csv` in the project root directory



## 3. Run Experiments


python experiments.py
```
## 🛠️ Features & Tech Stack

**Core Engine:** Python 3.x, NumPy  
**Data Handling:** Pandas  
**Visualization:** Matplotlib  

### Architecture

- Model: Multi-Layer Perceptron (MLP)
- Input: 784 neurons (28×28 pixels)
- Hidden layers: Configurable (e.g. `[128, 64]`)
- Output: 10 neurons (Softmax probabilities)

### Techniques

- Activation: ReLU (hidden), Softmax (output)
- Optimization: Stochastic Gradient Descent (SGD)
- Loss: Categorical Cross-Entropy



## 📂 Project Structure
----------
- Image-Recognition/
- ├── src/
- │   ├── model.py          # NeuralNetwork class & backprop engine
- │   ├── utils.py          # Batching, one-hot encoding, accuracy
- │   ├── initializers.py   # He uniform initialization
- │   ├── activations.py    # ReLU, Softmax & derivatives
- │   └── losses.py         # Cross-entropy loss
- ├── results/              # Generated training plots
- ├── train.py              # Training loop
- ├── experiments.py        # Experiment runner
- ├── requirements.txt
- └── README.md

## 🌍 Results
----------

| Metric              | Performance |
|---------------------|-------------|
| Validation Accuracy | ~96%        |
| Training Loss       | < 0.15      |
| Epochs              | 20          |
| Architecture        | 784 → 128 → 64 → 10 |

Results are based on a train/validation split of the MNIST training set.  
Exact performance may vary depending on initialization and hyperparameters.

* * *

## 🧠 What I Learned
----------------

* How backpropagation works at a mathematical level
* Why softmax and cross-entropy gradients simplify
* The impact of weight initialization on training stability
* How to structure ML projects cleanly without frameworks

## 👤 Author
--------

High school student learning artificial intelligence and machine learning.