---
title: Overfitting and Underfitting using Higher Order Linear Regression
subtitle: Exploring the concepts of Overfitting and Underfitting in Polynomial
  Regression using Python
date: 2021-11-11T17:09:34.681Z
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
#### Introduction and Objective

Our goal is explore and explain the concepts of model overfitting and underfitting using Higher Order Linear Regression. We are going to use Python modules PyTorch, NumPy and matplotlib to make and plot the results of our model. We would generate the data, divide it into train and test datasets, and then analyze it. The detailed problem statement can be found on the [Google Doc](https://docs.google.com/document/d/126p_RE60XSdpzmNWmO-OUOGsUbR-LczyYDQ9I4xnYAQ/edit#) by made my professor, [Dr. Deokgun Park](https://crystal.uta.edu/~park/), for the class CSE 5331 Data Mining. 

**Overfitting** is said to occur when a model corresponds very closely to training data set and usually contains more parameters than required by the data. The model generated works very well against training data and has a very high accuracy against it but not very accurate against any other unseen data. We can improve the model accuracy by *decreasing* the complexity of the model.

![Figure 1 : Image showing Overfit, Underfit and Optimal Model. Image Credit : https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/](image-1-overfitting-underfitting.png "Figure 1 : Image showing Overfit, Underfit and Optimal Model. Image Credit : https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/")

**Underfitting** is the opposite of Overfitting as the name suggests. It occurs when the model is too simple. The model generated is unable to capture the relationship between input and output variables accurately and has a high error rate for both training and test dataset. Its accuracy can be improved by making the model *more complex*.

#### Process

First, we import the required libraries.

```
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
```

###### a. Generate 20 data pairs (X, Y) using y = sin(2\*pi\*X) + 0.1 * N

* Using uniform distribution between 0 and 1 for X
* Sampling N from the normal gaussian distribution
* Using 10 for train and 10 for test

```
np.random.seed(0)

x = np.random.uniform(0.0, 1.0, 20)
N = np.random.normal(0.0, 1.0, 20)
y = np.sin(2*np.pi*x) + (0.1 * N)

x_train = x[:10]
x_test = x[10:]

y_train = y[:10]
y_test = y[10:]
```

###### b. Using root mean square error, find weights of polynomial regression for order is 0, 1, 3, 9

Root Mean Square Error (**RMSE**) is a metric that tells us the average distance between the predicted values from the model and the actual values in the dataset.

<span style="color: #000000;"><strong>RMSE = </strong>√<span style="border-top: 1px solid black;">Σ(P<sub>i</sub> – O<sub>i</sub>)<sup>2</sup> / n</span></span>

Where 

* Σ is sum
* P<sub>i</sub> is the predicted value for the i<sup>th</sup> observation in the dataset
* O<sub>i</sub> is the observed value for the i<sup>th</sup>  observation in the dataset
* n is the sample size



#### Links

1. [Our Jupyter Notebook with Code](https://github.com/skbt/Overfitting-in-polynomial-regression/blob/main/Overfitting-using-Higher-Order-Linear-Regression.ipynb)

#### References

1. https://www.pythonpool.com/matplotlib-table/
2. https://datascience.stackexchange.com/questions/80868/overfitting-in-linear-regression
3. https://en.wikipedia.org/wiki/Overfitting
4. https://www.ibm.com/cloud/learn/overfitting
5. https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/
6. https://www.ibm.com/cloud/learn/underfitting
7. https://www.geeksforgeeks.org/polynomial-regression-from-scratch-using-python/
8. https://www.statology.org/mean-squared-error-python/