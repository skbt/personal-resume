---
title: How to classify images using Convolutional Neural Network
subtitle: Creating a simple convolution neural network (CNN) to classify images
  from CIFAR-10 data set using PyTorch
date: 2021-11-03T00:32:26.651Z
draft: false
featured: false
tags:
  - Convolutional Neural Network
  - CNN
  - PyTorch
  - Deep Learning
  - Kaggle
  - CIFAR-10
  - Image Classification
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
###### Introduction and Objective

Our objective is to make a neural network model, which can correctly classify a given image into one of the 10 given classes of the CIFAR-10 dataset. We are using PyTorch library to make our model. 

**Convolution Neural Network (CNN)** are a feed-forward type of deep learning neural network. They are used to classify the data and are extensively used in making image classifiers, computer vision and video processing, audio and speech analysis, etc. The main difference in CNN and other matrix multiplication based neural networks is that a CNN uses *convolutions* in at least one layer of the neural network, i.e. takes it takes two functions and returns a function instead of using matrix multiplications, which become exponentially complex as the size of the data increases.

**CIFAR-10** dataset is a collection of 60000 32x32 labeled images divided into 10 classes, airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks, having 6000 images in each class. This dataset is one of the most commonly used dataset for making machine learning image classification algorithms and models. We would use this dataset for making our model. The dataset can be downloaded from the kaggle link provided in the links below.

**PyTorch** is an open-source machine learning library created by Facebook. It can be used for various applications such as natural language and audio/speech processing, computer vision etc. Another similar library is TensorFlow. In this project we are going to use PyTorch.

We would take the help of [PyTorch CIFAR-10 tutorial](<1. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py>) for making the classifier and try to make improvements on it.

###### Process

First we would have to import the torch package and download and load the data.

```
import torch
import torchvision
import torchvision.transforms as transforms
```

After downloading and uploading the CIFAR-10 data images, we then divide the dataset into train and test sets.

```
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

We get the following output

```
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
Extracting ./data/cifar-10-python.tar.gz to ./data
Files already downloaded and verified
```



###### Links

1. [CIFAR-10 Dataset](https://www.kaggle.com/c/cifar-10/data)
2. [Our Jupyter Notebook with code](https://github.com/skbt/CIFAR-10-classfier/blob/main/DataMining_Assignment1-CIFAR-10_Classifier.ipynb)

###### References

1. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
2. https://www.kaggle.com/c/cifar-10
3. https://deepai.org/machine-learning-glossary-and-terms/convolutional-neural-network
4. https://en.wikipedia.org/wiki/CIFAR-10
5. https://www.ibm.com/cloud/learn/convolutional-neural-networks
6. https://en.wikipedia.org/wiki/Convolutional_neural_network
7. https://en.wikipedia.org/wiki/PyTorch