# Deep Learning for Fashion Item Images Classification (Fashion-MNIST)

This project implements a deep learning system in Python using PyTorch to classify clothing items from the Fashion-MNIST dataset. It includes three models built from scratch:

- **Multilayer Perceptron (MLP)**
- **Convolutional Neural Network (CNN)**
- **Vision Transformer-style model**

To improve efficiency, **PCA (Principal Component Analysis)** was applied for input dimensionality reduction, accelerating MLP training. The workflow covers **data preprocessing, model training with the Adam optimizer, hyperparameter tuning, and performance evaluation** across all models.

Please refer to `report.pdf` for full details of our work.

## How to Run

Make sure the data is placed in the `dataset/` directory, that need to be created by extracting the dataset.zip file in its current location. And then run the code :

```bash
python main.py
```
