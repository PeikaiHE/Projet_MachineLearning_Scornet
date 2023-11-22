"""第一步，进行数据的初步了解"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')
path = "/Users/hepeikai/Library/CloudStorage/OneDrive-个人/Master_ISDS_Sorbonne/S3/Apprentissage Statistique_/data/"

data = pd.read_csv(path+"train.csv")

data.info()

missing_values = data.isnull().sum()
print(missing_values)
data_describe = data.describe()
print(data.columns)
# plot 'avg_glucose_level' and 'bmi'

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.distplot(data['avg_glucose_level'])
plt.subplot(1, 2, 2)
sns.distplot(data['bmi'])
plt.show()
