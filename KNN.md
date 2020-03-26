<!-- GFM-TOC -->
* [基于KNN的分类模型](#基于KNN的分类模型)
<!-- GFM-TOC -->

### 基于KNN的分类模型

为了判断未知样本的类别，以所有已知类别的样本作为参照，计算未知样本与所有已知样本的距离，从中选取与未知样本距离最近的K个已知样本，根据少数服从多数的投票法则，将未知样本与K个最邻近样本中所属类别占比较多的归为一类。

#### 算法简述


```
算法：knn
输入：训练集Dt，验证集Dv，超参k
输出：验证集数据的标签

for 验证集Dv中每个对象 Vd do

    for 训练集Dt中每个对象 Vt do
        计算对象Vd到对象Vt的距离d
    end for

    选出距离最小的前k个训练集对象构成序列Vote
    统计序列Vote中出现的最多类别lc
    并将lc赋予对象Vd
```

#### 算法实现

此处利用鸢尾花数据集使用python进行算法的实现

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import  KNeighborsClassifier
import numpy as np


def main(k,X_train,y_train,X_test,y_test):

    def distance(X_train,vec_t): # 欧氏距离

        diff_m = X_train - np.array([vec_t] * len(X_train))
        power_m = np.power(diff_m,2)
        return np.power(np.add.reduce(power_m,1),0.5)

    y_pred = []
    for vec_t in X_test:
        
        distance_m = distance(X_train,vec_t)
        k_max = np.argpartition(distance_m,k)[k:] # 取出距离最小的前k个
        # 获取label
        label_dict = {}
        for i in range(k):
            label = y_train[k_max[i]]
            try:
                label_dict[label] += 1
            except:
                label_dict[label] = 1
        maxV = max(list(label_dict.values()))
        for key,value in label_dict.items():
            if value == maxV:
                y_pred.append(key)
                break

    return y_pred



if __name__ == '__main__':

    X,y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=33)
    y_pred = main(5,X_train,y_train,X_test,y_test)
    print(classification_report(y_test,y_pred))
    y_pred = KNeighborsClassifier().fit(X_train,y_train).predict(X_test)
    print(classification_report(y_test,y_pred))



```

#### 实验效果与分析

- 自己实现的KNN

Precision | Recall | F1
---|---|---
0.96 | 0.96 | 0.95

- sklearn的KNN

Precision | Recall | F1
---|---|---
0.98 | 0.99 | 0.98

