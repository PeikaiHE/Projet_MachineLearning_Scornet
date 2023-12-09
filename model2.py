"""
首先用深度学习来预测Unknown可以如何分配给其他的三种吸烟情况。

之后组成更完整的数据之后可以使用传统的机器学习方法进行预测，也可以使用深度学习预测
"""
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
from tqdm import tqdm
from time import sleep
from sklearn.utils.class_weight import compute_class_weight




path = "/Users/hepeikai/Library/CloudStorage/OneDrive-个人/Master_ISDS_Sorbonne/S3/Apprentissage Statistique_/data/"

data_train = pd.read_csv(path+"train.csv")
data_test = pd.read_csv(path+"test.csv")

data_train = data_train[data_train['gender'] != 'Other']

"""print(data_train['smoking_status'].value_counts())
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
plt.show()"""

# 从上面的图表中可以看出，Unknown的值似乎可以被其他的三个条件解释
# 其中work type = children都是never smoked, 但是其他情况我们还不知道，所以第一步先把这一部分的Unknown定义为never smoked

data_train.loc[(data_train['smoking_status'] == 'Unknown') & (data_train['work_type'] == 'children'), 'smoking_status'] = 'never smoked'

# 之后通过机器学习来进行预测Unknown的值

# 分离出 'smoking_status' 已知和未知的数据
known_smoking_status = data_train[data_train['smoking_status'] != 'Unknown']
unknown_smoking_status = data_train[data_train['smoking_status'] == 'Unknown']

# 分离出特征和标签，特征里还要删除掉 'smoking_status' 和'stroke'
X_known = known_smoking_status.drop(['smoking_status', 'stroke'], axis=1)
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
X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=0.3, random_state=42)

"""# 进行模型的选择
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
results_df = pd.DataFrame(results).transpose()"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 应用预处理
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# 如果 y 是分类标签，使用 LabelEncoder 对其编码
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

X_train_tensor = torch.tensor(X_train_transformed).float()
y_train_tensor = torch.tensor(y_train_encoded).long()
X_test_tensor = torch.tensor(X_test_transformed).float()
y_test_tensor = torch.tensor(y_test_encoded).long()

# 创建 Dataset 和 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 定义模型
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout = nn.Dropout(0.2)
        self.l2 = nn.Linear(128, 128)
        self.l = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l(x))
        x = self.fc4(x)
        return x

# 输入和输出的大小
input_size = X_train_transformed.shape[1]
output_size = len(torch.unique(y_train_tensor))

model = Net(input_size, output_size)

"""model.eval()

# 关闭梯度计算
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)"""

# 计算类别权重
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# 应用类别权重
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# 定义损失函数和优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 降低学习率

min_val_loss = np.inf
patience = 0
num_epochs = 100
early_stopping_patience = 50
losses = []
accuracies = []
# 训练模型
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 后向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 计算验证损失
    val_loss = 0.0
    # 早停逻辑
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        patience = 0
    else:
        patience += 1
        if patience >= early_stopping_patience:
            print(f"Stopping early at epoch {epoch + 1}")
            break
    #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    losses.append(loss.item())
    accuracies.append((outputs.argmax(1) == labels).float().mean())

# 画出"Cross-entropy" 和 Accuracy 的图表
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(patience), losses[-patience - 1:])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
plt.plot(range(patience), accuracies[-patience - 1:])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# 将模型设置为评估模式
model.eval()

# 初始化一个列表来存储预测标签
predicted_indices = []

# 关闭梯度计算
with torch.no_grad():
    for inputs in DataLoader(X_test_tensor, batch_size=64):
        # 前向传播来获取模型的预测结果
        outputs = model(inputs)
        # 获取最大概率的索引，这是预测的类别标签
        _, predicted = torch.max(outputs, 1)
        # 将预测标签添加到列表中
        predicted_indices.extend(predicted.tolist())

# 将预测标签转换回原始的类别名称
predicted_labels = label_encoder.inverse_transform(predicted_indices)

# 评估模型
print(classification_report(y_test, predicted_labels))

# 使用模型来预测'Unknown'的值
X_unknown_transformed = preprocessor.transform(X_unknown)
X_unknown_tensor = torch.tensor(X_unknown_transformed).float()

# 将模型设置为评估模式
model.eval()

# 预测未知数据
predicted_indices_unknown = []
with torch.no_grad():
    outputs = model(X_unknown_tensor)
    _, predicted_unknown = torch.max(outputs, 1)
    predicted_indices_unknown.extend(predicted_unknown.tolist())

predicted_labels_unknown = label_encoder.inverse_transform(predicted_indices_unknown)

# 将预测的标签添加到原始数据中
data_train_corrected = data_train.copy()
data_train_corrected.loc[data_train_corrected['smoking_status'] == 'Unknown', 'smoking_status'] = predicted_labels_unknown

# 检查是否还有Unknown的值
data_train_corrected['smoking_status'].value_counts()

# 保存数据
data_train_corrected.to_csv(path + 'train_corrected.csv', index=False)