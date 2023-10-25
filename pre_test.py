"""第一步，进行数据的初步了解"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('./Projet_MachineLearning_Scornet/optiver-trading-at-the-close/train.csv')

data.info()

missing_values = data.isnull().sum()

data_describe = data.describe()

plt.figure(figsize=(12, 6))
sns.histplot(np.log(data['imbalance_size']), bins=100, kde=True) #这里的np.log是为了让数据更加平滑
plt.title('Distribution of Imbalance Size')
plt.xlabel('Imbalance Size')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

correlation_matrix = data.corr()
target_correlation = correlation_matrix["target"].sort_values(ascending=False)