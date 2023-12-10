"""
使用深度学习所得到的数据（train_corrected.csv和test_corrected.csv）进行建模
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
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


path = "/Users/hepeikai/Library/CloudStorage/OneDrive-个人/Master_ISDS_Sorbonne/S3/Apprentissage Statistique_/data/"

data_train = pd.read_csv(path + "train_corrected.csv")
data_test = pd.read_csv(path + "test_corrected.csv")
y_real = pd.read_csv(path + "sample_submission.csv")
print(data_train.columns)
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
    "Logistic Regression L1": LogisticRegression(penalty='l1', C = 1.0,solver='liblinear', max_iter=1000),
    "Logistic Regression L2": LogisticRegression(penalty='l2', C = 1.0,solver='lbfgs', max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0),
    "Gradient Boosting Machine": GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=0),
    "Extreme Gradient Boosting": XGBClassifier(n_estimators=50, learning_rate=0.1, random_state=0),
    "Support Vector Machine": SVC(gamma='auto'),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
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

# 找到最好的模型
results_df.to_csv(path + "results_model_NN.csv")

# 使用Support Vector Machine
model = SVC(gamma='auto')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 使用官方结果进行测试
X_real = data_test.drop(['id'], axis=1)
X_real_preprocessed = preprocessor.transform(X_real)
y_real_pred = model.predict(X_real_preprocessed)
print(sum(y_real_pred) / len(y_real_pred))
print(y_real['stroke'][0])