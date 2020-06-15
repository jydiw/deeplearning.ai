---
title: "DLS: CNN Case Studies"
excerpt_separator: "<!--more-->"
categories:
  - python
  - data science
tags:
  - deeplearning.ai
  - neural networks
  - convolutional neural networks
---

# practical advice for using convnets

## using open source implementations

search through github for previous implementations

`git clone {url}`
`cd {dir}`
`dir`

# transfer learning

because datasets can be so large and because it can take weeks to train very deep neural networks, perhaps it is better to use transfer learning (unless you have an exceptionally large dataset)

- ImageNet
- MS COCO
- Pascal

consider a classification problem between two cats: `tigger`, `misty`, or `neither`.

you may not have that many pictures of your cats, so your training set will be small.

download an open-source implentation of a computer vision network and their weights. if they have different output classes (say 1000 different ones), remove the softmax layer and replace with your own.

"freeze" the parameters of all the layers of the network and only train the softmax layer. you might be able to get pretty good performance with a small dataset.

because all of the early layers are frozen, you could find some fixed function that could directly feed into the last layer. that is, you can precompute all of the layers and compute a feature vector from your input, turning it into a shallow softmax network.

or: you could remove the last few layers and train your own hidden layers.

general rule of thumb: larger your training set, the fewer layers you'd have to freeze

for very large dataset, you could train the whole network using their published weights as initialization.

## data augmentation

common methods:
- mirroring (on vertical axis)
- random cropping (so long as crops are reasonably large subsets)
- color shifting (adding to / subtracting from certain channels) (PCA color augmentation)

less common methods:
- rotation
- shearing
- local warping

implementing distortions during training:
- hdd -> cpu thread -> implement distortion

# the state of computer vision

consider a spectrum between "little data" and "lots of data"

speech recognition probably around 0.6, whereas image recognition 0.4 and object detection 0.15

with lots of data, we tend to get away with simpler algorithms and less hand-engineering

with little data, we need to be more deliberate with hand-engineering to get better performance

two sources of knowledge:
- labeled data
- hand-engineered features / network architecture / other components

computer vision is quite complex, so we rely on hand engineering and the absence of more data

## tips for doing well on benchmarks

- ensemble methods
  - train several networks independently and average their **outputs** (NOT their weights)
- multi-crop at test time
  - run classifier on multiple versions of test images and average the results
  - 10-crop (center, four corners, mirror)
-