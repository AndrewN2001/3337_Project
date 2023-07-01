import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from matplotlib import rcParams
rcParams['figure.figsize'] = 15, 5
sns.set_style('darkgrid')

df_60s = pd.read_csv('dataset-of-60s.csv')
print(df_60s.head())
print(df_60s.info())

df_60s.drop(labels=['track', 'artist', 'uri'], axis=1, inplace=True) # We do not need this as it will not affect our results.
print(df_60s.info())

# Data Cleanup
# for track, group in df_60s.groupby('track'):
#     if len(group) > 1:
#         if group['target'].nunique() != 1 and group['artist'].nunique() != len(group):
#             print(group)
## Above is my attempt at cleaning up duplicate tracks that have different target values with the same artists.
## None of the songs have them, so it's perfectly clean.

print(df_60s.columns)

sns.countplot(x=df_60s['target'])
plt.show()



