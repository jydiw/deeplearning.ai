---
title: "DLS: Pooling and Fully-Connected Layers"
excerpt_separator: "<!--more-->"
categories:
  - python
  - data science
tags:
  - deeplearning.ai
  - neural networks
  - convolutional neural networks
---

# Max Pooling

if we have a 4x4 matrix and pool into four 2x2 regions, that is like having hyperparameters f = 2 and s = 2

essentially: if this feature is detected anywhere in this filter, then keep the high number.

caveat: it has been found to work well but not sure if anyone knows the real underlying reason of why it works (Ng)

example 2: 5x5 matrix using max pooling where f = 3 and s = 1, output is 3x3

3D input is same as before, number of channels remains the same with each iteration

Average Pooling

not really used as often. maybe used very deep in a NN to collapse your representation down the road

Note: no parameters to learn!

# fully-connected layers

convention: layers are counted by weight

LOOK UP GOOD WAYS TO VISUALIZE NEURAL NETWORKS

A fully-connected layer is just like a single neural network layer. Called fully-connected since each of the neurons from the Conv output is connected to the neurons in the FC layer.

Look into the literature for choosing hyperparameters.

Usually, nH and nW will decrease with each layer, and nC will increase with each layer. FC layers are usually at the end.

Most parameters come from the FC layers.

Activation size gradually decreases with each layer. If the size decreases too drastically, usually does not have great performance.

