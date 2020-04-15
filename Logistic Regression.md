<!-- GFM-TOC -->
* [二分类逻辑回归](#二分类逻辑回归)
* [多分类逻辑回归](#二分类逻辑回归)
* [参考文献](#参考文献)
<!-- GFM-TOC -->

```python
%matplotlib inline
import tensorflow as tf
from matplotlib import pyplot as plt
import random
```

## 二分类逻辑回归

### 模型定义

利用线性模型可以完成回归学习的任务，可若要进行二分类任务，则需要找一个单调可微函数将分类任务的真实标记y以及线性回归模型的预测值联系起来。为此，可以利用对数几率函数，即模型定义为


<center><div align=center><img src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/LogR-f1.png"/></div></center>

可以看出对数几率函数将z值转换为一个接近0或1的y值，经过计算，式子转换为

<center><div align=center><img src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/LogR-f2.png"/></div></center>

若将y视作x为正例的概率，y/(1-y)可视作对于x输入的正例相对几率。

### 数据集

随机生成一个用于二分类的二维数据集

```python
from numpy.random import RandomState
#通过随机函数生成一个模拟数据集
rdm = RandomState(1)
# 定义数据集的大小以及维度
dataset_size = 1000
x_feature_num = 2
# 模拟输入是一个二维数组
X = rdm.rand(dataset_size,x_feature_num).astype(np.float32)
#定义输出值，将x1+x2 < 1的输入数据定义为正样本
Y = np.array([[int(x1+x2 < 1)] for (x1,x2) in X]).astype(np.float32)

plt.figure(figsize=(8,4))
plt.xlabel('x')
plt.ylabel('y')
for i in range(X.shape[0]):
    if X[i,0] + X[i,1] < 1:
        plt.scatter(X[i,0],X[i,1], 1,marker='v',c='b')
    else:
        plt.scatter(X[i,0],X[i,1], 1,marker='v',c='r')
```
<center><div align=center><img src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/LogR-1.png"/></div></center>

### 模型训练

<center><div align=center><img src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/LogR-f3.png"/></div></center>

对于包含θ关于x的函数，可定义目标函数，以梯度下降的方式优化模型权重，训练得到模型

> 图片省略b权重仅是为了使得运算简便。事实上w和b可视作对x的参数，可以合并。代码中依然保留b权重

#### 定义模型

定义模型的前向传播过程,即逻辑回归的公式

```python
# sigmoid函数即对数几率函数y = 1/(1 + exp (-x))
def logreg(X,w,b):
    return tf.sigmoid(tf.matmul(X,w)+b)
```

#### 目标函数

目标函数有两类定义方法，一类使用常见的交叉熵函数，另一类由最大似然估计推导得到，本质上的含义是一致的

##### 交叉熵

二分类中所需的损失函数应该是这样的
1. 当样本标签类型为正，若逻辑回归函数计算出的标签为正时，损失函数值应为0，若逻辑回归函数计算出的标签为负，损失函数值应该为很大。
2. 当样本标签类型为负，若逻辑回归函数计算出的标签为负时，损失函数值应为0，若逻辑回归函数计算出的标签为正，损失函数值应该为很大。

因此，可以使用交叉熵函数

<center><div align=center><img src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/LogR-2.png"/></div></center>

##### 最大似然函数

hθ(x)可用于表示正样本的概率，则1-hθ(x)可用于表示负样本的概率，则有：

<center><div align=center><img src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/LogR-f4.png"/></div></center>
最大化l(θ)等价于最小化-l(θ)，即为目标函数

可以发现两种方法推导得到的目标函数是一致的


```python
def binary_cross_entropy(y_pred,y_true):
    # 将预测值压缩在一定范围之间，防止出现log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    return -tf.reduce_mean(y_true * tf.math.log(y_pred)+(1-y_true)*(tf.math.log(1-y_pred)))

bce = tf.keras.losses.BinaryCrossentropy()
```

#### 优化算法

利用小批量随机梯度下降可优化权重，利用tensorflow可省略具体计算
 - 非正则化权重为
<center><div align=center><img src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/LogR-3.png"/></div></center>
 - 正则化权重为
<center><div align=center><img src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/LogR-4.png"/></div></center>

```python
def sgd(params, lr, grads):
    """Mini-batch stochastic gradient descent."""
    for i, param in enumerate(params):
        param.assign_sub(lr * grads[i])

from tensorflow.keras import optimizers
trainer = optimizers.SGD(learning_rate=0.03)
```

#### 训练过程

##### 准备数据并定义超参

```python
lr = 0.03 # 学习率
num_epochs = 50 
batch_size = 10

w_m = tf.Variable(tf.random.normal((x_feature_num, 1), stddev=0.01))
b_m = tf.Variable(tf.zeros((1,)))

from tensorflow import data as tfdata

# 将训练数据的特征和标签组合
dataset = tfdata.Dataset.from_tensor_slices((X, Y))
# 随机读取小批量
dataset = dataset.shuffle(buffer_size=dataset_size) # 随机排列数据
dataset = dataset.batch(batch_size)
```

##### 模型训练过程

```python
from sklearn.metrics import classification_report

net = logreg
loss = binary_cross_entropy # Method 1
# loss = bce # Method 2

for epoch in range(num_epochs):
    for (batch, (batch_x, batch_y)) in enumerate(dataset):
        with tf.GradientTape() as t:
            t.watch([w_m,b_m])
            l = loss(net(batch_x, w_m, b_m), batch_y)
        grads = t.gradient(l, [w_m, b_m])
        sgd([w_m, b_m], lr, grads) # Method 1
#         trainer.apply_gradients(zip(grads, [w_m, b_m])) # Method 2
    train_l = loss(net(X, w_m, b_m), Y)
    Y_pred = tf.squeeze(net(X,w_m,b_m)).numpy()
    Y_pred = [1 if i >=0.5 else 0 for i in Y_pred]
    print(classification_report(tf.squeeze(Y),tf.cast(Y_pred,tf.int64)))
    print('epoch %d, loss %f' % (epoch, tf.reduce_mean(train_l)))
```

最终两个模型分类效果还是非常不错的
```
              precision    recall  f1-score   support

         0.0       0.92      1.00      0.96       536
         1.0       1.00      0.90      0.95       464

    accuracy                           0.95      1000
   macro avg       0.96      0.95      0.95      1000
weighted avg       0.96      0.95      0.95      1000
```

## 多分类逻辑回归

## 参考文献
- https://blog.csdn.net/u011734144/article/details/79717470
- 《统计学习方法》李航
- 《机器学习》周志华