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
#### Introduction and Objective

Our objective is to make a neural network model, which can correctly classify a given image into one of the 10 given classes of the CIFAR-10 dataset. We are using PyTorch library to make our model. 

**Convolution Neural Network (CNN)** are a feed-forward type of deep learning neural network. They are used to classify the data and are extensively used in making image classifiers, computer vision and video processing, audio and speech analysis, etc. The main difference in CNN and other matrix multiplication based neural networks is that a CNN uses *convolutions* in at least one layer of the neural network, i.e. takes it takes two functions and returns a function instead of using matrix multiplications, which become exponentially complex as the size of the data increases.

**CIFAR-10** dataset is a collection of 60000 32x32 labeled images divided into 10 classes, airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks, having 6000 images in each class. This dataset is one of the most commonly used dataset for making machine learning image classification algorithms and models. We would use this dataset for making our model. The dataset can be downloaded from the kaggle link provided in the links below.

**PyTorch** is an open-source machine learning library created by Facebook. It can be used for various applications such as natural language and audio/speech processing, computer vision etc. Another similar library is TensorFlow. In this project we are going to use PyTorch.

We would take the help of [PyTorch CIFAR-10 tutorial](<1. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py>) for making the classifier and try to make improvements on it.

#### Process

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

```
Files already downloaded and verified
Files already downloaded and verified
```

We can now test to see some of the images we downloaded.

```
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
```

We get the following random images

![](image-1-test-images.png)

We also get their classes.

```
truck   cat   car plane
```



###### Making our Neural Network

Now after testing the downloaded images, we can get to making our Convolutional Neural Network. We are going to use Classification Cross-Entropy loss for our function.

```
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```

```
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```



We can now start training our network.

```
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

We get the following output.

```
[1,  2000] loss: 2.208
[1,  4000] loss: 1.867
[1,  6000] loss: 1.694
[1,  8000] loss: 1.575
[1, 10000] loss: 1.529
[1, 12000] loss: 1.481
[2,  2000] loss: 1.399
[2,  4000] loss: 1.373
[2,  6000] loss: 1.353
[2,  8000] loss: 1.330
[2, 10000] loss: 1.319
[2, 12000] loss: 1.274
Finished Training
```

After training we can save our model so we can use it again without having to train it again.

```
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```

###### Testing our Neural Network

Now that we have made the neural network, we are going to test it's performance. To do this, we would make a prediction of the class the neural network is going to output and check it against the [ground-truth](https://en.wikipedia.org/wiki/Ground_truth#Statistics_and_machine_learning)

Output: 

![](image-2-ground-images.png)

```
GroundTruth:    cat  ship  ship plane
```

Now that we have established the ground truth, we can check what our neural network thinks these images are. Then we can also calculate the accuracy of the model on the whole dataset. To do that, first we can reload our model if it is not already loaded.

```

# Letting the neural network classify the testing data set
outputs = net(images)

# Getting the index of the highest energy
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


# Testing to see how the network performs on the whole dataset
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

Output:

```
Predicted:    cat  ship  ship plane
Accuracy of the network on the 10000 test images: 57 %
```

Though in our testing, we got our ground-truth and prediction in a small sample as 100% correct, when we used the whole dataset, we got the accuracy as 57%.



#### Contribution to improve the Model Accuracy

We got the accuracy of 57%. Our job to experiment with the model to improve its accuracy.

###### Experiment Model 1: Increase number of nodes in Convolution layer

We increase the number of to nodes in Convolution layer 50/100

```
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 50, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(50, 100, 5)
        self.fc1 = nn.Linear(100 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

```

We get the following output

```
[1,  2000] loss: 2.075
[1,  4000] loss: 1.697
[1,  6000] loss: 1.521
[1,  8000] loss: 1.419
[1, 10000] loss: 1.322
[1, 12000] loss: 1.279
[2,  2000] loss: 1.155
[2,  4000] loss: 1.123
[2,  6000] loss: 1.077
[2,  8000] loss: 1.052
[2, 10000] loss: 1.061
[2, 12000] loss: 0.991
Finished Training
```

We can see that our loss has reduced.

```
Accuracy of the network on the 10000 test images: 64 %
```

After establishing ground-truth and calculating accuracy, we find that our accuracy is **64%**, which is more than the base of 57%.

###### Experiment Model 2: Increasing number of epochs

Taking the previous code, where we increased the number of nodes, this time around we also increase number of epochs for model to train from 2 to 10.

```
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

We get the output

```
[1,  8000] loss: 1.407
[1, 10000] loss: 1.337
[1, 12000] loss: 1.279
[2,  2000] loss: 1.158
[2,  4000] loss: 1.116
[2,  6000] loss: 1.070
[2,  8000] loss: 1.058
[2, 10000] loss: 1.026
[2, 12000] loss: 1.009
[3,  2000] loss: 0.860
[3,  4000] loss: 0.865
[3,  6000] loss: 0.875
[3,  8000] loss: 0.860
[3, 10000] loss: 0.857
[3, 12000] loss: 0.852
[4,  2000] loss: 0.697
[4,  4000] loss: 0.700
[4,  6000] loss: 0.746
[4,  8000] loss: 0.736
[4, 10000] loss: 0.745
[4, 12000] loss: 0.734
[5,  2000] loss: 0.577
[5,  4000] loss: 0.600
[5,  6000] loss: 0.627
[5,  8000] loss: 0.611
[5, 10000] loss: 0.643
[5, 12000] loss: 0.637
[6,  2000] loss: 0.474
[6,  4000] loss: 0.502
[6,  6000] loss: 0.525
[6,  8000] loss: 0.528
[6, 10000] loss: 0.537
[6, 12000] loss: 0.546
[7,  2000] loss: 0.391
[7,  4000] loss: 0.399
[7,  6000] loss: 0.457
[7,  8000] loss: 0.454
[7, 10000] loss: 0.465
[7, 12000] loss: 0.470
[8,  2000] loss: 0.314
[8,  4000] loss: 0.347
[8,  6000] loss: 0.366
[8,  8000] loss: 0.422
[8, 10000] loss: 0.409
[8, 12000] loss: 0.404
[9,  2000] loss: 0.258
[9,  4000] loss: 0.282
[9,  6000] loss: 0.326
[9,  8000] loss: 0.325
[9, 10000] loss: 0.338
[9, 12000] loss: 0.354
[10,  2000] loss: 0.224
[10,  4000] loss: 0.285
[10,  6000] loss: 0.251
[10,  8000] loss: 0.286
[10, 10000] loss: 0.301
[10, 12000] loss: 0.306
Finished Training
```

We see that our loss has reduced considerably. 

```
Accuracy of the network on the 10000 test images: 72 %
```

After establishing ground truths and calculating accuracy we find that that our model is **72%** accurate and we have reduced the loss from 1.274 to 0.306.

![](image-4-final-improvement.png)

#### Links

1. [CIFAR-10 Dataset](https://www.kaggle.com/c/cifar-10/data)
2. [Our Jupyter Notebook with code](https://github.com/skbt/CIFAR-10-classfier/blob/main/DataMining_Assignment1-CIFAR-10_Classifier.ipynb)

#### References

1. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
2. https://www.kaggle.com/c/cifar-10
3. https://deepai.org/machine-learning-glossary-and-terms/convolutional-neural-network
4. https://en.wikipedia.org/wiki/CIFAR-10
5. https://www.ibm.com/cloud/learn/convolutional-neural-networks
6. https://en.wikipedia.org/wiki/Convolutional_neural_network
7. https://en.wikipedia.org/wiki/PyTorch