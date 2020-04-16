<!-- GFM-TOC -->
* [Softmax回归](#Softmax回归)
* [参考文献](#参考文献)
<!-- GFM-TOC -->

```python
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
```


## Softmax回归

逻辑回归亦可用于多分类任务上，有两种方法
- One vs Rest
    - 对于多个类别，将某一类视作正类，其它视作负类，构建多个分类器。
    - 适用于类别之间相互独立的状况
- Softmax
    - 将Sigmoid函数转换为Softmax函数输出解决多分类问题
    - 本文将实现这一算法

### 模型定义

对于输入特征，Softmax回归将其映射到多个类别的概率值。在逻辑回归中，h(x)输出维度为1，代表的是为正类的概率值，而Softmax回归中，h(x)输出维度为类别数量，代表的是每个类别的概率值。

<center><div align=center><img src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/SoftR-f1.png"/></div></center>

```python
def softmax_regression(x,W,b):
    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(tf.matmul(x, W) + b)
```

#### 目标函数
<center><div align=center><img src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/SoftR-1.png"/></div></center>

上式中1{·}为示性函数，为该类别时，函数值为1，否则为0。上述的目标函数也称作：对数似然代价函数。在二分类情况下，退化为交叉熵函数。

```python
# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred),1))

cce = tf.keras.losses.SparseCategoricalCrossentropy()
```

#### 优化过程
利用梯度下降法优化权重

<center><div align=center><img src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/SoftR-2.png"/></div></center>

```python
def sgd(params, lr, grads):
    """Mini-batch stochastic gradient descent."""
    for i, param in enumerate(params):
        param.assign_sub(lr * grads[i])

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)
```

#### 训练过程

##### 准备数据并定义超参

使用mnist数据集进行实验

```python
# MNIST dataset parameters.
num_classes = 10 # 0 to 9 digits
num_features = 784 # 28*28

# Training parameters.
learning_rate = 0.01
training_steps = 1000
batch_size = 256
display_step = 50

# Prepare MNIST data.
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# Weight of shape [784, 10], the 28*28 image features, and total number of classes.
W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")
# Bias of shape [10], the total number of classes.
b = tf.Variable(tf.zeros([num_classes]), name="bias")
```

##### 模型训练过程

```python
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = logistic_regression(x)
        loss = cross_entropy(pred, y)
        # loss = cce(y,pred)

    # Compute gradients.
    gradients = g.gradient(loss, [W, b])
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W, b]))
    # sgd([W, b], learning_rate, gradients) # Method 1

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0:
        pred = logistic_regression(batch_x)
        loss = cross_entropy(pred, batch_y)
        # loss = cce(batch_y,pred)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
```

## 参考文献
- https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/logistic_regression.ipynb