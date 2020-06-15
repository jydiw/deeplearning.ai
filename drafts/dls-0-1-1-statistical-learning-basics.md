---
title: "Statistical Learning Basics"
excerpt_separator: "<!--more-->"
categories:
  - data science basics
tags:
  - statistical learning
  -
---

# What Is Statistical Learning?

Statistical learning refers to a vast set of tools for understanding quantitative phenomena. Consider the following examples:

1. A client wishes to increase `sales` of a product by adjusting the advertising budgets for the product in three different media: `TV`, `radio`, and `newspaper`. Which budget increase would result in the highest increase in sales?
2. A commonly-held misconception is that owners of red cars tend to receive more speeding tickets than owners of other color cars--supposedly due to flashiness. Factoring in `age`, `gender`, & `location` of driver and `make`, `model`, & `color` of car, which factors contribute most to the receiving a `ticket`?

If increasing the `newspaper` advertising budget by $1000 only increased sales by $500, it might not be worth investing in that form of media. Or if the `color` of car had has no effect on the probability of receiving a ticket, perhaps you'd be more likely to purchase one yourself.

Here we attempt to predict the value of a **response** variable (`sales`, `ticket`) based on the values of one or more **predictors** (`TV`, `radio`, and `newspaper` advertising budgets, e.g.). By getting a better understanding of our data, we allow ourselves to make smarter decisions and/or uncover previously unknown relationships.

In general terms, we assume that there is some relationship $f$ between a response variable $Y$ and a set of $p$ predictors $X = \{X_1, X_2, \dots, X_p\}$, such that:

$$
Y = f(X) + \epsilon
$$

where $\epsilon$ represents a *random error* term.

## Types of Statistical Learning

Statistical learning can be split into two broad categories depending on the presence of an **output** variable:

- **Supervised** statistical learning: building a model to predict or estimate an *output* based on one or more *inputs* (such as in the two examples above)
- **Unsupervised** statistical learning: building a model to establish structure or relationships within the data with no supervising output

In addition, we can split depending on the nature of the response variable.

- **Continuous**: describing quantitative responses (e.g. `sales`)
- **Discrete**: qualitative or boolean responses (e.g. `ticket` $\in \{yes, no\}$)

<center>

| | Supervised | Unsupervised |
| ------------- |-------------| -----|
| **Continuous**| Regression | Dimensionality Reduction |
| **Discrete** | Classification | Clustering |

</center>

# Linear Regression

**Linear Regression** is a very simple approach for *supervised* learning and serves as the basis to some of the more advanced learning techniques used today. It attempts to estimate a linear relationship between one or more input variables and a *continuous* output variable.

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

- Each of the weights $w_i$ describe an algebraic relationship between a one-unit change in $x_i$ and its effect on $\hat{y}$.
- The intercept $b$ represents a baseline value when all independent variables have a value of $0$.

## Caveats

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