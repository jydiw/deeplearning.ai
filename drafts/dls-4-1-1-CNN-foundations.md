---
title: "DLS: Convolutional Neural Networks"
excerpt_separator: "<!--more-->"
categories:
  - python
  - data science
tags:
  - deeplearning.ai
  - neural networks
  - convolutional neural networks
---

A Convolutional Neural Network (CNN) is a multilayered neural network with a special architecture to detect complex features in data, with most applications in image classification, self-driving vehicles, and medical image analysis.

CNNs differ from multi-level perceptron networks (MLPs) by how regularization is applied. While the regularization of MLPs typically involve adjustments to how the loss function is calculated, CNNs take hierarchical data (such as that in images) and break complex patterns into smaller and simpler patterns, drastically reducing the number of learned parameters in the model. The result is faster computation and a lower likelihood of overfitting, even with highly complex datasets.

# The Problem with Detecting Cats

!!include pictures for each!!
- image classification
- object detection
- neural style transfer

One of the challenges of utilizing MLP networks for computer vision is that the inputs can be quite large. Let's consider training an MLP network to detect the presence of cats in a 1-megapixel 8-bit RGB image.

- 1-megapixel: the image contains 1 million pixels
- 8-bit RGB: there are three color channels (red, green blue) that can each have a value between 0-255.

If we were to feed this image as an input, we would have a total of 3 million input variables (1 million pixels per channel) for our input vector $\mathbf{x}$:

$$\mathbf{x} \in \mathbb{R}^{3 \times 1M}$$

If we were then feeding this into a 1000-neuron hidden layer, then our weight matrix $W$ would b a $(1000, 3M)$-dimensional matrix totalling 3 billion elements (and a bias term). Training 3 billion weights for each epoch (for one layer!) is far too computationally expensive for most computers to handle.

Let's take a step back and think about what "cat detection" means for us:

- a cat can be anywhere in the image for there to be positive detection
- there can be any number of cats in the image

This introduces the idea of **translational invariance**:
- the position of the object in an image should not be fixed to its position in order to be detected
- an identical set of features should result in identical predictive output even if its position is changed

Each pixel in an image represents a different input variable $x_n$

The question is: how can we reduce the computational complexity of a learning algorithm while simultaneously allowing translational invariance?

# Convolutional Neural Networks

Convolutional Neural Networks (CNNs) offer a solution to the computer vision problem by

- parameter sharing
- sparcity of connections

if all neurons are connected between a 32x32x3 and 28x28x6 layers, the weight matrix would be extremely large if fully connected

## parameter sharing:

a feature detector (such as a vertical edge detector) that's useful in one part of the image is probably useful in another part of the image, so the same parameter can be spread across the entire example

## sparsity of connections

In each layer, each output value depends only on a small number of inputs

This ultimately results in *translation invariance*, which means that an identical set of features should still result in the same output if it is shifted by a few pixels

## putting it all together

- start with image and separate by channel
- add feature detectors to create convolutional "volumes"
- add pooling lyers
- flatten volumes and add fully connected layers
- feed into softmax
- use gradient descent to minimize cost function J


# How Do Convolutional Neural Networks Work?

- Filters allow the algorithm to scan an image for spatial information
- Nearby pixels are more strongly related than distant ones
- Sets of features are aggregated to form the output
how to detect edges?

example with 6x6 matrix and convolve with 3x3 filter (sometimes called [kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing)))

(in mathematics, the asterisk is the standard operator for convolution)

conv-forward
tf.nn.conv2d
keras.conv2d

show example with 6x6 and a bunch of 10s

more filters:
- sobel filter
- scharr filter

deep learning can learn whatever filter it needs to detect edges

# padding

take 6x6 matrix and convolve with 3x3 filter you get 4x4 matrix

nxn matrix * fxf filter = (n-f+1) result

to solve shrinking output and throwing away info from edges / imbalanced weights

we can pad the image with an additional border. by convention we pad with 0s by a border of 1

"valid" : no padding
"same" : pad so output size is the same as input size

p = (f-1) / 2

by convention, f is almost always odd
- padding formula
- filter has a center

# strided convolutions

instead of moving by 1, move by stride $s$.

output is (n + 2p - f) / s + 1

if fraction is not integer, we take floor

# a note about convolution vs. cross-correlation

we've skipped mirroring operation and we're actually doing cross-correlation. in deep learning literature we just call it cross-correlation.



- [A Beginner's Guide to Convolutional Neural Networks]((https://heartbeat.fritz.ai/a-beginners-guide-to-convolutional-neural-networks-cnn-cf26c5ee17ed)])
- [Wikipedia: Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)