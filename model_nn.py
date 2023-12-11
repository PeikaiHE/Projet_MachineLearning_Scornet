"""
这个是使用深度网络的模型，而不是机器学习来进行预测
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout = nn.Dropout(0.2)
        self.l2 = nn.Linear(64, 128)
        self.l = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l(x))
        x = self.fc4(x)
        return x

path = "/Users/hepeikai/Library/CloudStorage/OneDrive-个人/Master_ISDS_Sorbonne/S3/Apprentissage Statistique_/data/"

data_train = pd.read_csv(path + "train_corrected.csv")
data_test = pd.read_csv(path + "test_corrected.csv")
y_real = pd.read_csv(path + "sample_submission.csv")

# 把数据分成X和y
X = data_train.drop(['id', 'stroke'], axis=1)
y = data_train['stroke']

# 创建预处理步骤
categorical_columns = X.select_dtypes(include=['object', 'bool']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

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

# 计算类别权重
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
print(class_weights)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# 应用类别权重
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# 定义损失函数和优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 降低学习率

min_val_loss = np.inf
patience = 0
num_epochs = 1000
early_stopping_patience = 400
losses = []
accuracies = []
val_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=128)
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

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_labels).item()
    val_loss /= len(val_loader)

    # 更新最小验证损失和耐心
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        patience = 0
        # 保存模型
        torch.save(model.state_dict(), path + 'model_new.pth')
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
plt.plot(range(patience+1), losses[-patience - 1:])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
plt.plot(range(patience+1), accuracies[-patience - 1:])
plt.xlabel('Epoch')
plt.ylabel('Accuracies')
plt.show()

# 将模型设置为评估模式
model.eval()
# 初始化一个列表来存储预测标签
predicted_indices = []

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

# 使用这个模型来预测data_test的数据
# 先进行OneHotEncoder处理
X_real = data_test.drop(['id'], axis=1)
X_real_transformed = preprocessor.transform(X_real)
X_real_tensor = torch.tensor(X_real_transformed).float()

# 将模型设置为评估模式
model.eval()

# 预测未知数据
predicted_indices_real = []
with torch.no_grad():
    outputs = model(X_real_tensor)
    _, predicted_real = torch.max(outputs, 1)
    predicted_indices_real.extend(predicted_real.tolist())

predicted_labels_real = label_encoder.inverse_transform(predicted_indices_real)
print(sum(predicted_labels_real) / len(predicted_labels_real)) # 0.04949524649612859