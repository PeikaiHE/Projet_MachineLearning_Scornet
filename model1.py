"""这一步我们删除了Unknown"""
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

data_train = pd.read_csv(path+"train.csv")
data_test = pd.read_csv(path+"test.csv")

data_train.info()

missing_values = data_train.isnull().sum()
print(missing_values)
print(data_train.describe())
print(data_train.columns)

# 列分析，首先分析性别
print(data_train['smoking_status'].value_counts())

sns.set(style="whitegrid")
plt.figure(figsize=(16, 6))  # 调整图表大小
plt.subplot(1, 3, 1)
sns.distplot(data_train['age'])
plt.title('Age Distribution')  # 添加子图标题
plt.subplot(1, 3, 2)
sns.distplot(data_train['avg_glucose_level'])
plt.title('Average Glucose Level Distribution')  # 添加子图标题
plt.subplot(1, 3, 3)
sns.distplot(data_train['bmi'])
plt.title('BMI Distribution')  # 添加子图标题
plt.suptitle('Distribution of Age, Average Glucose Level, and BMI', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()


# 删除不确定的值
data_train = data_train[data_train['smoking_status'] != 'Unknown']
data_train = data_train[data_train['gender'] != 'Other']

# 在确定了我们的数据库中没有异常值之后，我们就可以进行classification的建模了
# 分类特征和数值特征
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
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, data_train['stroke'], test_size=0.25, random_state=42)

# 初始化模型
# Régression logistique L1
log_reg_l1 = LogisticRegression(penalty='l1', C = 1.0,solver='liblinear', max_iter=1000)
log_reg_l1.fit(X_train, y_train)
y_pred = log_reg_l1.predict(X_test)
print(classification_report(y_test, y_pred))
# Régression logistique L2
log_reg_l2 = LogisticRegression(penalty='l2', C = 1.0,solver='lbfgs', max_iter=1000)
log_reg_l2.fit(X_train, y_train)
y_pred = log_reg_l2.predict(X_test)
print(classification_report(y_test, y_pred))
# Modèle Random Forest
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
# Modèle Gradient Boosting Machine
gbm = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=0)
gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)
print(classification_report(y_test, y_pred))
# Modèle Extreme Gradient Boosting
xgb = XGBClassifier(n_estimators=50, learning_rate=0.1, random_state=0)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print(classification_report(y_test, y_pred))
# SVM
svc = SVC(gamma='auto')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(classification_report(y_test, y_pred))
# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred))
# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print(classification_report(y_test, y_pred))
# Neural Network
nn = MLPClassifier(max_iter=1000)
nn.fit(X_train, y_train)
y_pred = nn.predict(X_test)
print(classification_report(y_test, y_pred))
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
results_df.to_csv(path + 'model_performance_results.csv', index=False)

# 在经过分析之后，我们决定使用XGBoost进行建模
param = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 10],
}
xgb1 = XGBClassifier(random_state=2)
grid_search = GridSearchCV(xgb1, param, cv=5, scoring='recall')
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_
print('Best parameters:', best_parameters) #Best parameters: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50}
print('Best model score:', grid_search.best_score_) #Best model score: 0.9522924411400246

# 找到了最佳的参数，我们就可以使用最佳的参数进行建模了
xgb2 = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=10, random_state=2)
xgb2.fit(X_train, y_train)
y_pred = xgb2.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Accuracy:{accuracy:.3f}') #Accuracy:0.948
print(f'Precision:{precision:.3f}') #Precision:0.899
print(f'Recall:{recall:.3f}') #Recall:0.948
print(f'F1 Score:{f1:.3f}') #F1 Score:0.923
# 计算错误率
misclassified_count = len(y_test[y_test != y_pred])
total_cases = len(y_test)
error_rate = misclassified_count / total_cases * 100
print(f"{misclassified_count} misclassified cases out of {total_cases}, error rate : {round(error_rate,2)}%")

# 接下来我们使用官方的测试集进行测试
X_test1 = data_test[data_test['smoking_status'] != 'Unknown']
test_id = X_test1['id']
X_test1 = X_test1.drop(['id'], axis=1)
y_test1 = pd.read_csv(path + 'sample_submission.csv')
y_test1 = y_test1[y_test1['id'].isin(test_id)]
print(sum(y_test1['stroke']) / len(y_test1['stroke']))
X_test_preprocessed1 = preprocessor.transform(X_test1)
y_pred1 = xgb2.predict(X_test_preprocessed1)
print(np.count_nonzero(y_pred1))
print(sum(y_pred1) / len(y_pred1))
# 我tm怎么发现差距这么多？ 是不是因为把unknown的值给去掉了？