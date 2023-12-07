#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 01:02:38 2023

@author: chenyuxiang
"""


# 完整的代码，包括从文件加载数据、预处理、绘图以及保存图表

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 从文件加载数据
train_data = pd.read_csv('train.csv')

# 预处理：标准化数值列，编码分类列
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_train = preprocessor.fit_transform(train_data.drop('stroke', axis=1))
y_train = train_data['stroke']

# 将处理后的数据转换为 DataFrame，以便进行绘图
X_train_df = pd.DataFrame(X_train, columns=preprocessor.get_feature_names_out())
X_train_df['stroke'] = y_train

# 绘制图表
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# 分布图
sns.histplot(data=X_train_df, x='num__age', hue='stroke', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution de l\'Âge')

sns.countplot(data=train_data, x='hypertension', hue='stroke', ax=axes[0, 1])
axes[0, 1].set_title('Distribution de l\'Hypertension')

sns.countplot(data=train_data, x='heart_disease', hue='stroke', ax=axes[1, 0])
axes[1, 0].set_title('Distribution de la Maladie Cardiaque')

sns.histplot(data=X_train_df, x='num__avg_glucose_level', hue='stroke', kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Distribution du Niveau Moyen de Glucose')

sns.histplot(data=X_train_df, x='num__bmi', hue='stroke', kde=True, ax=axes[2, 0])
axes[2, 0].set_title('Distribution de l\'IMC')

sns.countplot(data=train_data, x='smoking_status', hue='stroke', ax=axes[2, 1])
axes[2, 1].set_title('Distribution du Statut Tabagique')
axes[2, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()

# 保存图表
image_path = 'analyzed_data_visualization.png'
fig.savefig(image_path)
plt.close(fig)  # Close the figure to prevent it from displaying in the output

# 输出保存的图表路径
image_path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# L2正则化的逻辑回归模型
logreg_l2 = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
logreg_l2.fit(X_train, y_train)

# 预测测试集
y_pred_l2 = logreg_l2.predict(X_test)

# 打印分类报告
print("L2 Regularized Logistic Regression classification report:")
print(classification_report(y_test, y_pred_l2))

# L1正则化的逻辑回归模型
logreg_l1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=1000)
logreg_l1.fit(X_train, y_train)

# 预测测试集
y_pred_l1 = logreg_l1.predict(X_test)

# 打印分类报告
print("L1 Regularized Logistic Regression classification report:")
print(classification_report(y_test, y_pred_l1))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 训练随机森林模型
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
report_rf = classification_report(y_test, y_pred_rf)

# 训练梯度提升机模型
gbm_model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
gbm_model.fit(X_train, y_train)
y_pred_gbm = gbm_model.predict(X_test)
report_gbm = classification_report(y_test, y_pred_gbm)

# 训练极端梯度提升模型
xgb_model = XGBClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
report_xgb = classification_report(y_test, y_pred_xgb)

print("Random Forest Model Classification Report:")
print(report_rf)

print("\nGradient Boosting Machine Model Classification Report:")
print(report_gbm)

print("\nExtreme Gradient Boosting Model Classification Report:")
print(report_xgb)
