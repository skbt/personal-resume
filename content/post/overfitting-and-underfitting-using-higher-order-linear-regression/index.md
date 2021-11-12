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
from prettytable import PrettyTable
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

###### b. Using root mean square error, finding weights of polynomial regression for order is 0, 1, 3, 9

**Gradient Descent** is an optimization algorithm which is commonly-used to train machine learning models and neural networks. It works by finding a local minimum of a differentiable function. It takes repeated steps in the opposite direction of the gradient (or approximate gradient) of the function at the current point. **Learning rate** is the size of the steps that are taken to reach the minimum. The method of gradient descent updates the value of the weights based on the derivative of the loss function with respect to the weight.

Root Mean Square Error (**RMSE**) is a metric that tells us the average distance between the predicted values from the model and the actual values in the dataset.

<span style="color: #000000;"><strong>RMSE = </strong>√<span style="border-top: 1px solid black;">Σ(P<sub>i</sub> – O<sub>i</sub>)<sup>2</sup> / n</span></span>

Where 

* Σ is sum
* P<sub>i</sub> is the predicted value for the i<sup>th</sup> observation in the dataset
* O<sub>i</sub> is the observed value for the i<sup>th</sup>  observation in the dataset
* n is the sample size

We are going to find and update weight using W_new = W_old - 0.01*dL/dW, where the constant 0.01 is the learning rate dictating the impact of the gradient on the new weight. This method of gradient descent is applied for each and every weight, w, of the model.



```
degree = [0, 1, 3, 9]
model = []
for d in degree:
  m = Model(d)
  model.append(m.train())


class Model(object):

  def __init__(self, pdegree):
    self.pdegree = pdegree
    self.w = []
	self.w0 = torch.ones(1, requires_grad = True)
	self.w.append(self.w0)
	if self.pdegree > 0:
		for i in range(self.pdegree):
			self.w.append(torch.ones(1, requires_grad = True))

  def forward(self, x):
	num = self.w0
	if self.pdegree > 0:
		for i in range(1, len(self.w)):
			num = num + self.w[i]*x**i
    return num

  def loss(self, x_train, y_train):
    y_pred = self.forward(x_train)
    return (y_pred - y_train) * (y_pred - y_train)

  def train(self, epoch = 10):
    print()
    print('For polynomial of degree ' + str(self.pdegree))
    for i in range(epoch):
      lossvalue = 0
      for j in range(len(x_train)):
        l = loss(x_train[j], y_train[j])
        lossvalue += l
      lossvalue.backward()
      print('Iteration:', i, 'Loss:', lossvalue.data.item())
      self.w0.data = self.w0.data - 0.01*self.w0.grad.data
      if self.pdegree > 0:
		for k in range(1, len(self.w)):
			self.w[k].data = self.w[k].data - 0.01*self.w[k].grad.data

    
    print('w0 :', self.w0.item())
    if self.pdegree > 0:
      for k in range(1, len(self.w)):
		print('w' + str(k) + ' :', self.w[k].item())

    M = [self.w0.item()]
    if self.pdegree > 0:
      for i in range(1, len(self.w)):
        M.append(self.w[i].item())
 
    return M
```

Using epoch =10, we get the following output

```

For polynomial of degree 0
Iteration: 0 Loss: 17.66655921936035
Iteration: 1 Loss: 12.206013679504395
Iteration: 2 Loss: 5.434935569763184
Iteration: 3 Loss: 2.49934458732605
Iteration: 4 Loss: 5.630291938781738
Iteration: 5 Loss: 12.448257446289062
Iteration: 6 Loss: 17.771587371826172
Iteration: 7 Loss: 17.554550170898438
Iteration: 8 Loss: 11.962095260620117
Iteration: 9 Loss: 5.244487285614014
w0 : -0.2020360231399536

For polynomial of degree 1
Iteration: 0 Loss: 37.99821472167969
Iteration: 1 Loss: 21.088102340698242
Iteration: 2 Loss: 4.740396499633789
Iteration: 3 Loss: 5.836953639984131
Iteration: 4 Loss: 23.216598510742188
Iteration: 5 Loss: 38.87610626220703
Iteration: 6 Loss: 36.582923889160156
Iteration: 7 Loss: 18.657501220703125
Iteration: 8 Loss: 3.575340747833252
Iteration: 9 Loss: 6.8678436279296875
w0 : 1.006404161453247
w1 : 0.40355294942855835

For polynomial of degree 3
Iteration: 0 Loss: 76.13774871826172
Iteration: 1 Loss: 35.21417999267578
Iteration: 2 Loss: 4.1375603675842285
Iteration: 3 Loss: 21.363895416259766
Iteration: 4 Loss: 65.25962829589844
Iteration: 5 Loss: 81.0052490234375
Iteration: 6 Loss: 48.74273681640625
Iteration: 7 Loss: 8.278226852416992
Iteration: 8 Loss: 9.639623641967773
Iteration: 9 Loss: 50.92899703979492
w0 : 1.8113633394241333
w1 : 0.7055118680000305
w2 : 0.39220136404037476
w3 : 0.36708498001098633

For polynomial of degree 9
Iteration: 0 Loss: 179.9287109375
Iteration: 1 Loss: 70.34640502929688
Iteration: 2 Loss: 7.7664079666137695
Iteration: 3 Loss: 80.0475082397461
Iteration: 4 Loss: 180.31773376464844
Iteration: 5 Loss: 161.70538330078125
Iteration: 6 Loss: 49.623294830322266
Iteration: 7 Loss: 5.504806995391846
Iteration: 8 Loss: 94.158203125
Iteration: 9 Loss: 190.01898193359375
w0 : 2.8707337379455566
w1 : 0.969760000705719
w2 : 0.25171470642089844
w3 : 0.01874123513698578
w4 : -0.019307851791381836
w5 : 0.017044663429260254
w6 : 0.07815936207771301
w7 : 0.14408139884471893
w8 : 0.20724961161613464
w9 : 0.2652101218700409
```



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
9. https://www.ibm.com/cloud/learn/gradient-descent
10. https://en.wikipedia.org/wiki/Gradient_descent