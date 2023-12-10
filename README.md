# Projet_MachineLearning 
**E.Scornet**

- Kaggle 链接: https://www.kaggle.com/competitions/playground-series-s3e2/data
- Overleaf 链接：https://www.overleaf.com/8326179925tkwfmcgpythz#a0551b

## Projet 结构

### 主要的结构
#### 第一个模型
1. 使用去掉 `smoking_status == 'Unknown' ` 的数据集直接进行建模。 我之前已经使用了这些模型：
`models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(max_iter=1000)
}`

并且通过这些模型，找到了最好的模型。之后再对这个模型进行`GridSearchCV`，找到最好的参数进行建模 (准确率高达95%）。 （已经完成）

2. 使用相同的模型，但是加上`smoking_status == 'Unknown' `的数据进行建模，结果是一团糟（与官方给的平均值相比较），接下来引出第二个模型. （还没做，得做要不然咋引出第二种模型）

#### 第二个模型

1. 将 `smoking_status == 'Unknown' ` 的数据进行建模，目的是将所有的Unknown分配到其他三种情况中。 之后使用这个更新的数据进行普通机器学习, 重复刚才的第一步。（已经完成，调参也差不多了）

2. 使用刚才数据集使用深度学习进行建模，和传统的机器学习进行比较。 （做了一点，在`estim_ml.py`里面）


#### 为什么我们的模型准确率没有非常高
我觉得是因为在classification的时候，我们的数据集是不平衡的，所以我们的模型会倾向于预测`stroke == 0`的情况，而不是`stroke == 1`的情况。
这就导致了在准确率等方面的反馈不错，但是和官方的数据相比有差距。

### 针对于各个python文件的解释
- `analyse.py` : 这个是陈宇翔的code
- `zhang_xu.py`, `test xu.ipynb` : 这个是张旭的code
- `model1.py` : 这个是我们project的第一步。使用去掉 `smoking_status == 'Unknown' ` 和 `gender == Other` 的数据集直接进行建模。 找到最好的模型，之后找到最好的param。
- `model2.py` : 使用深度学习针对`smoking_status == 'Unknown' `进行建模，得到一个修改过的`train_corrected.csv`，之后使用这个数据集进行普通机器学习。
- `DNN.py` : 这个是使用`model2.py`中的模型对`test.py`中的数据进行处理，得到一个修改过的`test_corrected.csv`，之后使用这个数据集进行普通机器学习。
- `estim_ml.py` : 这一步使用的是经过`model2.py`&`DNN.py`的深度网络处理过的数据集，找到最合适的机器学习模型（实际上就是重复了一下`model1.py`中机器学习的步骤）
- `model_weighted.py` : 这一步使用的是经过`model2.py`&`DNN.py`的深度网络处理过的数据集，之后再使用`estim_ml.py`中的模型进行建模。 得到的模型已经不错了
- `model_nn.py` : 这一步使用的是经过`model2.py`&`DNN.py`的深度网络处理过的数据集，之后使用深度学习进行建模。 得到的模型已经也不错了
