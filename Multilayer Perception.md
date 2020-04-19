<!-- GFM-TOC -->
* [线性神经网络](#线性神经网络)
* [激活函数](#激活函数)
* [非线性神经网络](#非线性神经网络)
<!-- GFM-TOC -->

### 线性神经网络

单个神经元接收输入信号经过非线性变换得到输出，多个神经元在横向以及纵向上组合可以应用于复杂的任务中。
<center><div align=center><img src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/MLP-1.png"/></div></center>

具体来说，对于上层的网络，设输入特征为X，隐藏层特征为H，输出层特征为O，则有

<center><div align=center><img src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/MLP-f1.png"/></div></center>

这样的式子即便在网络的宽度和深度增加后也仅能表达输入特征与输出类别的线性关系。


### 激活函数

问题的根源在于全连接层只是对数据做仿射变换（affine transformation），而多个仿射变换的叠加仍然是一个仿射变换。解决问题的一个方法是引入非线性变换，例如对隐藏变量使用按元素运算的非线性函数进行变换，然后再作为下一个全连接层的输入。这个非线性函数被称为激活函数（activation function）。下面介绍常用的三种激活函数

##### Relu
<center><div align=center><img height=100,width=200, src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/MLP-f2.png"/></div></center>
<center><div align=center><img height=150,width=250, src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/MLP-2.png"/></div></center>
##### sigmoid
<center><div align=center><img height=100,width=200, src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/MLP-f3.png"/></div></center>
<center><div align=center><img height=150,width=250, src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/MLP-3.png"/></div></center>
##### tanh
<center><div align=center><img height=100,width=200, src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/MLP-f4.png"/></div></center>
<center><div align=center><img height=150,width=250, src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/MLP-4.png"/></div></center>


### 非线性神经网络
定义激活函数为o(·)，则非线性神经网络有如下下定义:

<center><div align=center><img src ="https://github.com/Teren-Liu/Machine-Learning/blob/master/image/MLP-f5.png"/></div></center>
且Python实现为:

```python
W1 = tf.Variable(tf.random.normal(shape=(num_inputs, num_hiddens),mean=0, stddev=0.01, dtype=tf.float32))
b1 = tf.Variable(tf.zeros(num_hiddens, dtype=tf.float32))
W2 = tf.Variable(tf.random.normal(shape=(num_hiddens, num_outputs),mean=0, stddev=0.01, dtype=tf.float32))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=0.1))

def relu(x):
    return tf.math.maximum(x,0)
    
def net(X):
    # X.shape = (batch_size,num_inputs)
    H = relu(tf.matmul(X, W1) + b1)
    return tf.math.softmax(tf.matmul(H, W2) + b2)
```