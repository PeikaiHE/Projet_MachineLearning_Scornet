"""这一步我们不能删除Unknown的值了，因为之前所得到的结果不好"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


path = "/Users/hepeikai/Library/CloudStorage/OneDrive-个人/Master_ISDS_Sorbonne/S3/Apprentissage Statistique_/data/"

data_train = pd.read_csv(path+"train.csv")
data_test = pd.read_csv(path+"test.csv")

data_train = data_train[data_train['gender'] != 'Other']

print(data_train['smoking_status'].value_counts())
# 我想知道在smoking_status中，Unknown可以被怎样的三个值解释，也可以检验一下其他三个值中中风的数量
# 先看看其他三个值中中风的数量，用plot体现出来
sns.set(style="whitegrid")
plt.figure(figsize=(16, 6))  # 调整图表大小
plt.subplot(1, 3, 1)
sns.countplot(x='smoking_status', hue='stroke', data=data_train)
plt.title('Stroke Compté par Smoking Status')  # 添加子图标题
plt.subplot(1, 3, 2)
sns.countplot(x='smoking_status', hue='ever_married', data=data_train)
plt.title('Marriage Compté par Smoking Status')  # 添加子图标题
plt.subplot(1, 3, 3)
sns.countplot(x='smoking_status', hue='work_type', data=data_train)
plt.title('Work Type Compté par Smoking Status')  # 添加子图标题
plt.suptitle('Plots de compté de Smoking Status', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

# 从上面的图表中可以看出，Unknown的值似乎可以被其他的三个条件解释
# 其中work type = children都是never smoked, 但是其他情况我们还不知道，所以第一步先把这一部分的Unknown定义为never smoked

data_train.loc[(data_train['smoking_status'] == 'Unknown') & (data_train['work_type'] == 'children'), 'smoking_status'] = 'never smoked'

# 之后通过机器学习来进行预测Unknown的值

# 分离出 'smoking_status' 已知和未知的数据
known_smoking_status = data_train[data_train['smoking_status'] != 'Unknown']
unknown_smoking_status = data_train[data_train['smoking_status'] == 'Unknown']

# 分离出特征和标签
X_known = known_smoking_status.drop(['smoking_status'], axis=1)
y_known = known_smoking_status['smoking_status']
X_unknown = unknown_smoking_status.drop(['smoking_status'], axis=1)

# 创建预处理步骤
categorical_columns = X_known.select_dtypes(include=['object', 'bool']).columns
numerical_columns = X_known.select_dtypes(include=['int64', 'float64']).columns

# 数值特征预处理
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# 分类特征预处理
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)])
# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=0.2, random_state=42)

# 进行模型的选择
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
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', model)])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # 评估模型并且记录
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = report
results_df = pd.DataFrame(results).transpose()

# 接下来使用Pytorch来进行模型的训练，我们要对X_train,X_test,y_train,y_test进行转换
X_train = OneHotEncoder(sparse=False).fit_transform(X_train)
X_test = OneHotEncoder(sparse=False).fit_transform(X_test)
y_train = LabelEncoder().fit_transform(y_train)
y_test = LabelEncoder().fit_transform(y_test)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 转换为 PyTorch tensors
X_train_tensor = torch.tensor(X_train.values).float()
y_train_tensor = torch.tensor(y_train.values).long()
X_test_tensor = torch.tensor(X_test.values).float()
y_test_tensor = torch.tensor(y_test.values).long()

# 创建 Dataset 和 DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(torch.unique(y_tensor)))  # 输出层的大小等于类别数

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 后向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


# 预测已知吸烟状态的测试集
y_pred = model.predict(X_test)

# 评估模型
print(classification_report(y_test, y_pred))

# 预测未知吸烟状态的数据
predicted_smoking_status = model.predict(X_unknown)
