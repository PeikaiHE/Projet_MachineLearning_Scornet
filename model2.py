"""这一步我们不能删除Unknown的值了，因为之前所得到的结果不好"""
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


path = "/Users/hepeikai/Library/CloudStorage/OneDrive-个人/Master_ISDS_Sorbonne/S3/Apprentissage Statistique_/data/"

data_train = pd.read_csv(path+"train.csv")
data_test = pd.read_csv(path+"test.csv")

data_train = data_train[data_train['gender'] != 'Other']

# 我想知道在smoking_status中，Unknown可以被怎样的三个值解释，也可以检验一下其他三个值中中风的数量
# 先看看其他三个值中中风的数量，用plot体现出来
sns.set(style="whitegrid")
plt.figure(figsize=(16, 6))  # 调整图表大小
plt.subplot(1, 3, 1)
sns.countplot(x='smoking_status', hue='stroke', data=data_train)
plt.title('Stroke Count by Smoking Status')  # 添加子图标题
plt.subplot(1, 3, 2)
sns.countplot(x='smoking_status', hue='ever_married', data=data_train)
plt.title('Marriage Count by Smoking Status')  # 添加子图标题
plt.subplot(1, 3, 3)
sns.countplot(x='smoking_status', hue='work_type', data=data_train)
plt.title('Work Type Count by Smoking Status')  # 添加子图标题
plt.suptitle('Count Plots of Smoking Status', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

# 从上面的图表中可以看出，Unknown的值在smoking_status

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
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, data_train['stroke'], test_size=0.2, random_state=42)
