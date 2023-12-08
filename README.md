# Projet_MachineLearning 
**E.Scornet**

- Kaggle 链接: https://www.kaggle.com/competitions/playground-series-s3e2/data
- Overleaf 链接：https://www.overleaf.com/8326179925tkwfmcgpythz#a0551b

## Projet 结构

### 主要的结构
- 使用去掉 `smoking_status == 'Unknown' ` 的数据集直接进行建模。 我之前已经使用了
`models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(max_iter=1000)
}`

找到最好的模型，之后在进行`GridSearchCV`，找到最好的参数进行建模。 （已经完成）

- 之后是将 `smoking_status == 'Unknown' ` 的数据进行建模，目的是将所有的Unknown分配到其他三种情况中。 之后使用这个更新的数据进行普通机器学习
重复刚才的步骤。（基本完成，只剩下调参数）

- 使用刚才更新的数据集使用深度学习进行建模。 （还没做）


