---
title: "DLS: Object Detection"
excerpt_separator: "<!--more-->"
categories:
  - python
  - data science
tags:
  - deeplearning.ai
  - neural networks
  - convolutional neural networks
---

In order to talk about object detection, we must start with object localization:

- classification
- classification with localization
- detection

# classification with localization

sample softmax output:
- 1 - pedestrian
- 2 - car
- 3 - motorcycle
- 4 - background

you can also have your network output four more numbers to define a "bounding box" of the detected object.

> upper left: (0,0)
> lower right (1,1)

$(b_x, b_y)$ is midpoint of bounding box
$(b_w, b_h)$ is width and height of bounding box

sample output vector:

$$
\mathbf{y} = \begin{bmatrix}
p_c \\
b_x \\
b_y \\
b_h \\
b_w \\
c_1 \\
c_2 \\
c_3
\end{bmatrix}
$$

where:

$p_c$ probability of object (not background) present in image
$c_i$ whether it belongs in any of the classes

Loss can then be calculated as:

$$loss(\hat{y}, y) =
\begin{cases}
\displaystyle \sum_{i=1}^{8}(\hat{y}_i- y_i)^2 & \quad \text{if} \ \ y_1 = 1\\
(\hat{y}_1- y_1)^2 & \quad \text{if}\ \ y_1 = 0
\end{cases}
$$

In practice, log feature loss for c_i, squarred error for bounding box, and log-loss for pc.

# landmark detection

You can modify the output label to detect other landmarks, such as faces

# object detection

sliding windows detection: start with small "window" of main image and feed that crop into convnet

repeat process with slightly larger window

huge disadvantage: computational cost.
so many different crops for a single image.

coarse stride size could reduce computation but reduce performance as you could skip over objects in the image

## convolutional implementation of sliding windows

instead of a fully connected layer, convolve using a filter with as many filters as neurons in your fc layer

