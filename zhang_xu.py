import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split


path = "/Users/hepeikai/Library/CloudStorage/OneDrive-个人/Master_ISDS_Sorbonne/S3/Apprentissage Statistique_/data/"

train_data = pd.read_csv(path + "train.csv")
test_data = pd.read_csv(path + "test.csv")
print(train_data.columns)
# Display the first few rows of each dataset for initial inspection
train_head = train_data.head()
test_head = test_data.head()


train_data.isnull().sum()

test_data.isnull().sum()

train_data.describe()

test_data.describe()

stroke_distribution = train_data['stroke'].value_counts(normalize=True)

stroke_distribution

train_data_corrected = train_data.copy()
unknown_indices = train_data_corrected[train_data_corrected['smoking_status'] == 'Unknown'].index
valid_smoking_statuses = train_data_corrected['smoking_status'].unique()
valid_smoking_statuses = valid_smoking_statuses[valid_smoking_statuses != 'Unknown']
random_smoking_statuses = np.random.choice(valid_smoking_statuses, size=len(unknown_indices))
train_data_corrected.loc[unknown_indices, 'smoking_status'] = random_smoking_statuses

train_data = train_data_corrected
train_data['smoking_status'].value_counts()

train_data.head()
train_data['work_type'].value_counts()
train_data['Residence_type'].value_counts()
train_data['ever_married'].value_counts()