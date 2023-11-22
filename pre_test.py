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

data_describe = data.describe()
