---
title: "Linear Regression"
excerpt_separator: "<!--more-->"
categories:
  - data science basics
tags:
  - statistical learning
  -
---

<center>

| | Supervised | Unsupervised |
| ------------- |-------------| -----|
| **Continuous**| **REGRESSION** | Dimensionality Reduction |
| **Discrete** | Classification | Clustering |

</center>

# Linear Regression

**Linear Regression** is a very simple approach for *supervised* learning and serves as the basis to some of the more advanced learning techniques used today. It attempts to estimate a linear relationship between one or more input variables and a *continuous* output variable.

Let's say we wanted to estimate a person's `weight` as it relates to their `height`. Here, the `weight` is the **response** variable (also called the **dependent variable**) which we want to predict, and their `height` is the **predictor** (also called the **independent variable**). We can read this as, "a person's weight is approximately modeled as a function of their height":

$$
\begin{array}{rcl}
y &\!\!\approx&\!\!b + w_1x\\
\mathtt{weight} &\!\!\approx&\!\! b + w_1 \cdot \mathtt{height}
\end{array}
$$

$b$ and $w_1$ are two unknown constants that represent the **intercept** and **slope** in our linear model.

- The slope $w_1$ describes an algebraic relationship between a one-unit change in $x$ and its effect on $\hat{y}$. e.g. How much does a 1-cm increase in `height` affect a person's expected `weight` in kg?
- The intercept $b$ describes a baseline value of $y$ when the value of $x$ is 0.

Once we have used our training data to produce estimates for this relationship, we can write:

$$
\begin{array}{rcl}
\hat{y} &\!\!=& \!\!\hat{b} + \hat{w}_1x
\end{array}
$$

Here, the *hat* symbol, $\hat{\;\;}$, denotes the estimated value for an unknown parameter or the predicted value of the response variable. $\hat{y}$ ("$y$ hat") is the predicted `weight`, $\hat{w}_1$ is the predicted slope for `height`, and $\hat{b}$ is the predicted intercept.

Let's use some actual data, taken from [here](https://helloacm.com/the-machine-learning-case-study-how-to-predict-weight-over-heightgender-using-linear-regression/).




Consider the linear regression equation:

$$
\begin{array}{rcl}
\hat{y} &=& \hat{b} + \hat{w}_1x_1 + \hat{w}_2x_2 + \cdots + \hat{w}_nx_n \\
&=& \hat{b} + \displaystyle \sum_{i=1}^{n} \hat{w}_ix_i
\end{array}
$$

>- $x_i$ -- the $i$th **independent variable**
>- $\hat{w}_i$ -- the $i$th **coefficient**
>- $\hat{b}$ -- the **intercept**
>- $\hat{y}$ -- the **predicted value** of our **dependent variable**



The linear regression is a highly interpretable model:

- Each of the
- The intercept $b$ represents a baseline value when all independent variables have a value of $0$.



However, the linear regression model only works for continuous variables. If we were training a model to categorize images as either `dog` and a `cat`, we could attempt to map this model in the following way:

$$y =
\begin{cases}
0 & \quad \text{if dog}\\
1 & \quad \text{if cat}
\end{cases}
$$

We know that no linear combination of cats would sum up to a dog (e.g. $\text{cat} - \text{cat} \neq \text{dog}$). Furthermore, values such as $0.5$, $1.2$, and $-0.3$ would not hold any real-world meaning (e.g. cat-dog, super-cat, or anti-dog). How do we confine our function to only output within the range $[0, 1]$?

> We only focus on range $[0, 1]$ since we previously defined the bounds to mean `dog` and `cat`, respectively. If we had defined our categories as $1$ and $2$ we would want to confine our model to the range $[1, 2]$.



The **logistic regression** is a popular classification algorithm that shares many properties with the highly interpretable and computationally lean linear regression model. As such, it is by far the most common classification algorithm.

# The Logistic Regression
**Logistic Regression** is a *supervised* statistical learning paradigm which adapts the linear regression model and maps to the *probability* of belonging to one of multiple categories:

$$\hat{y} = P(y = 1)$$

Thus, a value of $0.8$ would mean that the model predicts the example to be 80% a cat, 20% a dog.

In order to confine our function to range $[0, 1]$, we commonly use the *sigmoid function*:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

![sigmoid](./assets/images/sigmoid.png)

The function asymptotically approaches $1$ when reaching values of $+\infty$ (and $0$ at $-\infty$), allowing us to map all the real numbers in the desired range.

The logistic regression equation is just the linear regression output fed into the sigmoid function.

$$z =  \sum_{i=1}^{n} w_ix_i + b$$
$$\hat{y} = \sigma(z)$$

## Logistic Regression model optimization is achieved via the Logistic Loss function.

A simple least-squared error can't be used for a logistic regression since our output variable is no longer linear. Predicting $\hat{y} = 0.9$ means $P(y = 1) = 0.9$. This means that $y$ is predicted to be 9 times as likely to be $1$ than $0$. However, $\hat{y} = 0.8$ means $y$ is predicted to be 4 times as likely to be $1$ than $0$; this is because the odds.