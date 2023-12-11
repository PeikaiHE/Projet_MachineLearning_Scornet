"""
这个步骤中我想修改类权重，对少数类进行加权，使得少数类的权重大一些，这样可以使得模型更加关注少数。
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

data_train = pd.read_csv(path+"train_corrected.csv")
data_test = pd.read_csv(path+"test_corrected.csv")

# 对弱分类器进行加权
# 1. 对数据进行分割
X = data_train.drop(['id', 'stroke'], axis=1)
y = data_train['stroke']
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
X_preprocessed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, data_train['stroke'], test_size=0.25, random_state=42)

# 2. 进行加权
from sklearn.utils.class_weight import compute_class_weight
class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_test), y=y_test)
class_weight = dict(enumerate(class_weight))
print(class_weight)
class_weights = {0: 1, 1: 4.9}
print(data_train['smoking_status'].value_counts())
# 3. 进行训练 SVM
model = SVC(class_weight=class_weights,gamma='auto')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


X_test1 = data_test.copy()
test_id = X_test1['id']
X_test1 = X_test1.drop(['id'], axis=1)
y_test1 = pd.read_csv(path + 'sample_submission.csv')
y_test1 = y_test1[y_test1['id'].isin(test_id)]
print(sum(y_test1['stroke']) / len(y_test1['stroke'])) # 0.041296393099851524
X_test_preprocessed1 = preprocessor.transform(X_test1)
y_pred1 = model.predict(X_test_preprocessed1)
print(np.count_nonzero(y_pred1)) #417
print(sum(y_pred1) / len(y_pred1)) # 0.040870332255219056


