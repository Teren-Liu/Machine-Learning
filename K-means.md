<!-- GFM-TOC -->
* [基于K-means的聚类模型](#基于K-means的聚类模型)
* * [K-means聚类](#K-means聚类)
* * [二分K-means聚类](#二分K-means聚类)
<!-- GFM-TOC -->

### 基于K-means的聚类模型

为了利用K-means模型进行聚类，我们对聚类得到的簇进行定义,簇类包括以下方法：
- 静态方法distance(vec_a,vec_b): 获取两个向量间的欧氏距离
- 静态方法show_cluster(cluster_list,title): 对于多个簇以点图的形式显示
- add_node(node): 增加节点
- remove(node): 取出节点
- get_sse(): 计算该簇的sse值

```python
import numpy as np
import matplotlib.pyplot as plt
from math import *


class Cluster:

    def __init__(self):
        self.nodes = []
        self.centroid = None  # 该簇质心

    @staticmethod
    def distance(vec_a: np.array, vec_b: np.array) -> np.array:
        diff = vec_a - vec_b # X_i - Y_i
        return sqrt(np.dot(diff, diff)) # sqrt((X_i - Y_i)^2)

    @staticmethod
    def show_cluster(cluster_list,title='Cluster Process'):

        cnames = {'aliceblue': '#F0F8FF','antiquewhite': '#FAEBD7','aqua': '#00FFFF','aquamarine': '#7FFFD4','azure': '#F0FFFF'}  # color_dict

        for i, c in enumerate(cluster_list):
            a = np.array(c.nodes) # 对于每一个簇的节点建立二维矩阵
            plt.scatter(a[:, 0], a[:, 3], c=list(cnames.values())[i]) # 绘图

        plt.title(title)
        plt.show()

    def add_node(self,node: np.array) -> bool:

        if len(self.nodes) == 0:
            self.centroid = node
            self.nodes.append(node)
        else:
            self.centroid = (len(self.nodes) * self.centroid + node) / (len(self.nodes)+1)
            self.nodes.append(node)
        return True

    def remove_node(self, node: np.array) -> bool:

        remove_index = -1

        for i,n in enumerate(self.nodes):
            if type(node==n)==bool:
                remove_index = i
                break

        self.centroid = (len(self.nodes) * self.centroid - node) / (len(self.nodes) - 1)  # 更新簇心
        self.nodes.pop(remove_index)
        return True


    def get_sse(self) -> float:
        sse = 0
        for i in self.nodes:
            sse += pow(self.distance(self.centroid,i),2)
        return sse
```

### K-means聚类

#### 简介

k-means算法是1967年由MacQueen首次提出的一种经典算法，它是一种基于质心的划分方法，这种方法将簇中所有对象的平均值看作是簇的质心，根据一个数据对象与簇质心的距离，将该对象赋予最近的簇。在此类方法中，需要给定划分的簇个数k，首先得到k个初始划分的集合，然后采用迭代重定位技术，通过将对象从一个簇移到另一个簇来改进划分的质量。

#### 算法描述

```
算法：k-means
输入：数据集D，划分簇的个数k
输出：k个簇的集合
从数据集D中任意选择k个对象作为初始质心
repeat：
       for 数据集D中每个对象 P do
            计算对象P到k个簇中心的距离
            将对象P指派到与其最近的簇
       end for
        计算每个簇中对象的均值，作为新的簇中心
until  k个簇的簇中心不再发生变化
```

#### 代码实现

此处利用鸢尾花数据集使用python编码进行算法的实现

```python

from cluster import * # 上文定义的cluster类
from sklearn.datasets import load_iris
import numpy as np
import random

def main(k,data,show_process=True):
    cluster_list = []
    note = [-1] * len(data)
    cluster_change = True

    for i in range(k):
        c = Cluster()
        initCentroid = np.array(random.choice(data))
        c.add_node(initCentroid)
        cluster_list.append(c)

    while cluster_change == True:
        cluster_change = False
        for i, item in enumerate(data):
            minDist = float('inf')
            minIndex = -1
            for j in range(k):
                distance = Cluster.distance(cluster_list[j].centroid, item)
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            if note[i] != minIndex:
                cluster_change = True
                if note[i] != -1:
                    cluster_list[note[i]].remove_node(item)
                cluster_list[minIndex].add_node(item)
                note[i] = minIndex
        if show_process:
            Cluster.show_cluster(cluster_list,title='k-means')
            sse = 0
            for i in cluster_list:
                sse += i.get_sse()
            print(sse)
    return cluster_list


if __name__ == '__main__':

    iris = load_iris()
    data = iris.data
    k = 4
    main(k,data)




```

#### 实验效果与分析

![image](17B984183C3C4D0E9F49E5932D0507F1)

K-means算法描述容易，实现简单，快速，但存在以下不足：
1. k需要提前给定
1. 算法对初始值选取依赖性极大以及算法常陷入局部最优解
1. 离群点和噪声点会影响簇质心偏离
1. 不能处理分类属性的簇

#### 参考资料
- https://blog.csdn.net/llh_1178/article/details/81633396
- 《数据挖掘原理与实践》


### 二分K-means聚类

由于传统的KMeans算法的聚类结果易受到初始聚类中心点选择的影响，因此在传统的KMeans算法的基础上进行算法改进，对初始中心点选取比较严格，各中心点的距离较远，这就避免了初始聚类中心会选到一个类上，一定程度上克服了算法陷入局部最优状态。<br><br>
二分KMeans(Bisecting KMeans)算法的主要思想是：首先将所有点作为一个簇，然后将该簇一分为二。之后选择能最大限度降低聚类代价函数（也就是误差平方和）的簇划分为两个簇。以此进行下去，直到簇的数目等于用户给定的数目k为止。以上隐含的一个原则就是：因为聚类的误差平方和能够衡量聚类性能，该值越小表示数据点越接近于他们的质心，聚类效果就越好。所以我们就需要对误差平方和最大的簇进行再一次划分，因为误差平方和越大，表示该簇聚类效果越不好，越有可能是多个簇被当成了一个簇，所以我们首先需要对这个簇进行划分。<br><br>
二分K-means算法是基于层次的聚类算法

#### 算法描述


```
算法：二分k-means
输入：数据集D，划分簇的个数k，每次二分试验的次数m
输出：k个簇的集合
从数据集D中任意选择k个对象作为初始质心
repeat：
       从簇表中选取一个SSE最大的簇
       for i=1 to m do
            使用k-means算法对选定的簇聚类，划分为两个子簇；
       end for
       从m次二分试验所聚类的子簇中选择具有最小总SSE的两个簇;
       将这两个簇添加到簇表中；
until 簇表中包含k个簇
```

#### 代码实现

```python

from cluster import * # 上文提到的cluster
from sklearn.datasets import load_iris
import kmeans # 上文提到的kmeans

def main(k,m,data):

    cluster_list = []
    c_all = Cluster()
    show_process = True

    for i in data:
        c_all.add_node(i)
    cluster_list.append(c_all)

    while len(cluster_list) != k:

        maxSSE = 0
        minSSE = float('inf')

        for i in cluster_list:
            if i.get_sse() > maxSSE:
                maxSSE = i.get_sse()
                tmp_c = i

        cluster_list.remove(tmp_c)

        for i in range(m):

            tmp_list = kmeans.main(2, tmp_c.nodes,show_process=False)
            tmpSSE = sum(i.get_sse() for i in tmp_list)
            # 　选出生成最小ＳＳＥ的两个簇
            if tmpSSE < minSSE:
                min_cluster_list = tmp_list
                minSSE = tmpSSE

        cluster_list.extend(min_cluster_list)

        if show_process:
            Cluster.show_cluster(cluster_list, 'bi-kmeans')


if __name__ == '__main__':

    iris = load_iris()
    data = iris.data
    k = 4
    m = 4
    main(k,m,data)

```

#### 实验效果与分析

![image](5E0B5BE817884C249622FA4627EAC8E4)

可以发现，与K-means相比，二分K-means聚类效果更好，算法不再受初始化节点的影响，但算法花销也更大，对于K值以及m值需要多次调整才可获得更好的聚类效果。

#### 参考资料

- https://www.cnblogs.com/eczhou/p/7860435.html
- 数据挖掘原理与实践》