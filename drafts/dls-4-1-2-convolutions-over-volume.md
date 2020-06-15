---
title: "DLS: Convolutions Over Volume"
excerpt_separator: "<!--more-->"
categories:
  - python
  - data science
tags:
  - deeplearning.ai
  - neural networks
  - convolutional neural networks
---

Since color images have multiple channels per example, we can think of each input as a *volume*--that is, a three-dimensional matrix with `height`, `width`, and `channel` dimensions.

Consider a $100 \times 100$ RBG image, that means each example has $100 \times 100 \times 3$ features to be fed into the model.

Filter dimensions can be different, but the number of channels (also called 'depth' in the literature) must match the image channels.

eg: only detecting red vertical edges:

$$R = \begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix} \qquad G = \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
\end{bmatrix} \qquad B = \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
\end{bmatrix} $$

what if you want to use multiple filters?

output can then be stacked in 3D

Summary:

> For a layer $l$ in a neural network:
>
>- $f^{[l]}$ -- filter size of $l$th layer
>- $p^{[l]}$ -- padding size of $l$th layer
>- $s^{[l]}$ -- stride


$$
\begin{array}{lc}
  \text{input: \;}
  & n_H^{[l-1]} \times n_W^{[l-1]} \times n_c^{[l-1]}
  \\
  \text{filter: \;}
  & f_H^{[l]} \times f_W^{[l]} \times n_c^{[l-1]}
  \\
  \text{output: \;}
  & n_H^{[l]}\times n_W^{[l]} \times n_c^{[l]}
\end{array}$$

activations: $a^{[1]} = n_H^{[l]}\times n_W^{[l]} \times n_c^{[l]}$

weights: $f_H^{[l]} \times f_W^{[l]} \times n_c^{[l-1]} \times n_c^{[l]}$

bias: $n_c^{[l]} - (1,1,1,n_c^{[l]})$
where:
$$n^{[l]} = \left\lfloor\frac{n^{[l-1]}+2p^{[l]} - f^{[l]}}{s^{[l]}} + 1 \right\rfloor$$

# one layer of a CNN

recall that:

$$z^{[1]} = W^{[1]}a^{[0]} + b^{[1]}$$
where $a^{[1]} = g(z^{[1]})$

test: if you have 10 filters that are 3x3x3 in one layer of a neural network, how many parameters does that layer have?

3x3x3 = 27
+ bias = 28
* 10 filters = 280

# a simple example

let's say we have an image that has a resolution of 39x39x3 (RGB)

$$n_H^{[0]} \times n_W^{[0]} \times n_c^{[0]}$$
where
$$
n_H^{[0]} = n_W^{[0]} = 39 \\
n_c^{[0]} = 3
$$

If we use a set of ten 3x3 filters to detect features:

$$
f_H^{[1]} = f_W^{[1]} = 3\\
s^{[1]} = 1\\
p^{[1]} = 0
$$
Then our output layer will be 37x37x10:
$$a^{[1]} \in \mathbb{R}^{37 \times 37 \times 10}
$$
$$
n^{[1]} = \left\lfloor\frac{39+2\cdot0 - 3}{1} + 1 \right\rfloor = 37$$

If the next layer has twenty 5x5 filters with a stride of 2 and padding of 0:

$$
f_H^{[2]} = f_W^{[2]} = 5\\
s^{[2]} = 2\\
p^{[1]} = 0
$$

then:
$$a^{[2]} \in \mathbb{R}^{17 \times 17 \times 20}
$$
$$
n^{[2]} = \left\lfloor\frac{37+2\cdot0 - 5}{2} + 1 \right\rfloor = 17$$

Applying a set of 40 5x5 filters with a stride of 2 and padding of 0, we get:

$$a^{[3]} \in \mathbb{R}^{7 \times 7 \times 40}
$$

This means that we are asking the network to look at $7\times7\times40$, or $1960$ features which we could then unroll into an $\mathbb{R}^{1960\times1}$ vector fed into logistic or softmax.

## three types of layers
- convolution (conv)
- pooling (pool)
- fully-connected (fc)