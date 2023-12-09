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
