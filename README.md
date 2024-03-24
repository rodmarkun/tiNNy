# TiNNy: Simple neural networks built from scratch with just Python and Numpy

![ab3c57d8c8851397c82aa3868b60b611](https://github.com/rodmarkun/tiNNy/assets/75074498/d1f93819-5f40-4029-82d1-1bc6938f68b3)


TiNNy is a minimalist project aimed at demonstrating the power and potential of neural networks using nothing but Python and Numpy. This project is designed for educational purposes, allowing users to understand the fundamentals of neural network architectures, forward propagation, backpropagation, and training processes in a hands-on manner.

## Dive Into Demos

TiNNy comes equipped with two sample Python notebooks. These demos are a great way to see TiNNy in action:

- [demo_classification.ipynb](https://github.com/rodmarkun/tiNNy/blob/master/demo_classification.ipynb): TiNNy classification project using the MNIST handwritten digit dataset and a Heart attack prediction dataset.
- [demo_regression.ipynb](https://github.com/rodmarkun/tiNNy/blob/master/demo_regression.ipynb): TiNNy regression project using the Boston housing price and the Diamond price prediction datasets.

## Math explained

We begin by having an input of features X and an input of labels y, as neural networks are part of supervised learning. The goal is to learn a function that maps inputs to outputs, minimizing the error between the predicted outputs and the actual labels. This process involves several steps, prominently including the forward pass, backpropagation, and parameter update. Here's how each step works mathematically:

### Forward

In the forward pass, the network applies a series of transformations to the input data to compute the output predictions. This involves computing the linear combination of inputs and weights, adding a bias, and then applying an activation function. The process for layer i can be visualized as:

![equation1](https://github.com/rodmarkun/tiNNy/assets/75074498/6f3994a3-317d-4e9e-abc3-63a06c9f67f6)

Here, L^i represents the linear combination at layer i, w^i and b^i are the weights and bias for layer i, respectively, and A^(i-1) is the activation from the previous layer or the input data for the first layer. A^i is the output of the activation function applied to L^i, moving the data forward through the network.

### Backpropagation

Backpropagation is the process of adjusting the network's weights and biases to minimize the error between the actual labels and the predictions. It involves two key steps: computing the gradients for the output layer and then for the dense (hidden) layers.

#### Backpropagation in Output Layer

The gradient of the loss with respect to the output layer's weights, biases, and activations are computed as follows:

![equation2](https://github.com/rodmarkun/tiNNy/assets/75074498/123e1957-bad3-4c98-8a99-c9ae0f5a0a82)

dL^k denotes the derivative of the loss with respect to the activations of the last layer, dW^k and db^k are the gradients of the loss with respect to the weights and biases of the last layer, respectively. A^k is the activation of the last layer, and y is the true labels.

#### Backpropagation in Dense Layers

For each of the preceding layers, the gradients are computed taking into account the derivative of the activation function to propagate the error backward through the network:

![equation3](https://github.com/rodmarkun/tiNNy/assets/75074498/d181e5c1-4ed5-48ed-a3af-7aa14e02da61)

Here, g'(L^i) represents the derivative of the activation function applied at layer i, allowing the gradient of the loss to be propagated back through the network.

### Update Parameters

Finally, the parameters of the network are updated using the gradients computed during backpropagation. This step adjusts the weights and biases in the direction that minimally reduces the error:

![equation4](https://github.com/rodmarkun/tiNNy/assets/75074498/2c628d59-d9a9-44cd-8662-4d3071e6d2c3)

W^i and b^i are updated by subtracting a fraction of the gradients dW^i and db^i, scaled by a learning rate alpha. This iterative process of forward pass, backpropagation, and parameter update continues until the model's performance reaches a satisfactory level.

![Forward](https://github.com/rodmarkun/tiNNy/assets/75074498/339488c1-4917-4a99-bf00-83787918f5dd)

## Getting Started with TiNNy

Start using TiNNy by setting up the project on your personal computer. Follow these simple steps to get started:

First, clone the repository to your local machine using Git. Open your terminal and run the following commands:

```bash
git clone https://github.com/rodmarkun/tiNNy
cd tiNNy
```

TiNNy requires certain Python packages to function properly. Ensure all dependencies are installed by executing the following command in the terminal:

```bash
pip install -r ./requirements.txt
```

With the installation complete, you're ready to incorporate TiNNy into your Python scripts or Jupyter notebooks. Simply import the library using:

```python
import tinny
```

You are now equipped to explore the features and capabilities of TiNNy. Happy coding!
