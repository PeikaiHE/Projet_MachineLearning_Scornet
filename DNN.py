"""
使用训练好的神经网络对测试集进行预测
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

path = "/Users/hepeikai/Library/CloudStorage/OneDrive-个人/Master_ISDS_Sorbonne/S3/Apprentissage Statistique_/data/"

data_test = pd.read_csv(path + "test.csv")
data_test = data_test[data_test['gender'] != 'Other']

# 因为在data_test中的unknown还没有被分出来，所以我们进行model2.py中使用深度网络的方法，我们已经通过model2.py得到了训练好的模型，接下来我们直接使用就好了
# 首先我们需要对data_test进行预处理
# 分类特征和数值特征
known_smoking_status = data_test[data_test['smoking_status'] != 'Unknown']
unknown_smoking_status = data_test[data_test['smoking_status'] == 'Unknown']

# 分离出特征和标签，特征里还要删除掉 'smoking_status' 和'stroke'
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
X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=0.3, random_state=42)

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

# 读取训练好的模型
input_size = X_train_transformed.shape[1]
output_size = len(torch.unique(y_train_tensor))

model = Net(input_size, output_size)
model.load_state_dict(torch.load(path + 'model.pth'))
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

# 将预测标签添加到原始数据中
data_test_corrected = data_test.copy()
data_test_corrected.loc[data_test_corrected['smoking_status'] == 'Unknown', 'smoking_status'] = predicted_labels_unknown
data_test_corrected['smoking_status'].value_counts()
data_test_corrected.to_csv(path + 'test_corrected.csv', index=False)