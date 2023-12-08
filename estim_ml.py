"""
使用深度学习所得到的数据进行建模
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


path = "/Users/hepeikai/Library/CloudStorage/OneDrive-个人/Master_ISDS_Sorbonne/S3/Apprentissage Statistique_/data/"

data_train = pd.read_csv(path + "train_corrected.csv")
data_test = pd.read_csv(path + "test.csv")
y_real = pd.read_csv(path + "sample_submission.csv")

# 把数据分成X和y
X = data_train.drop(['id', 'stroke'], axis=1)
y = data_train['stroke']
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
# 构建预处理管道
# 对数值特征进行标准化
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# 对分类特征进行独热编码
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 合并处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 应用预处理并分割数据集
X_preprocessed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# 初始化模型
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(max_iter=1000)
}

# 训练模型并评估
results = {}
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 存储结果
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
# 填充DataFrame
for name, metrics in results.items():
    results_df = results_df.append({
        'Model': name,
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1 Score': metrics['F1 Score']
    }, ignore_index=True)

# 使用Logistic Regression进行建模
# 设置逻辑回归模型的参数网格
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # 正则化强度的倒数
    'penalty': ['l1', 'l2'],               # 使用的惩罚项
    'solver': ['liblinear', 'saga']        # 优化算法
}

# 配置网格搜索
log_reg = LogisticRegression(max_iter=1000,random_state=42)
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')

# 拟合网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数和最佳模型
best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_

# 输出最佳参数和最佳模型的得分
print('Best parameters:', best_parameters) #Best parameters: {'C': 0.001, 'penalty': 'l1', 'solver': 'liblinear'}
print('Best model score:', grid_search.best_score_) # Best model score: 0.9593203975478855
log_reg1 = LogisticRegression(max_iter=1000, C=0.001, penalty='l1', solver='liblinear', random_state=42)
log_reg1.fit(X_train, y_train)
y_pred = log_reg1.predict(X_test)
print(classification_report(y_test, y_pred)) # 96%的准确率

# 接下来我们使用官方的测试集进行测试
y_model = log_reg1.predict(X_preprocessed)

# 看看我们的模型的预测结果以及不同值的数量
unique, counts = np.unique(y_model, return_counts=True)
print("Our model's prediction:", dict(zip(unique, counts)))

print("Our model's mean is:", y_model.mean())

print("The real means is:", y_real['stroke'][0])
