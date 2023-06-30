import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

from matplotlib import rcParams
rcParams['figure.figsize'] = 15, 5
sns.set_style('darkgrid')

df_90s = pd.read_csv('dataset-of-90s.csv')
df_80s = pd.read_csv('dataset-of-80s.csv')
df_70s = pd.read_csv('dataset-of-70s.csv')
df_60s = pd.read_csv('dataset-of-60s.csv')
df_10s = pd.read_csv('dataset-of-10s.csv')
df_00s = pd.read_csv('dataset-of-00s.csv')
print(df_00s.head())
print(df_10s.info())
