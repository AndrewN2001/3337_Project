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

df = [pd.read_csv(f'dataset-of-{decade}0s.csv') for decade in ['6', '7', '8', '9', '0', '1']]
for i, decade in enumerate([1960, 1970, 1980, 1990, 2000, 2010]):
    df[i]['decade'] = pd.Series(decade, index=df[i].index)
data = pd.concat(df).reset_index(drop=True)

print(data.head())
print(data.info())

data.drop(labels=['track', 'artist', 'uri'], axis=1, inplace=True) # We do not need this as it will not affect our results.
print(data.info())

# Data Cleanup
# for track, group in df_60s.groupby('track'):
#     if len(group) > 1:
#         if group['target'].nunique() != 1 and group['artist'].nunique() != len(group):
#             print(group)
## Above is my attempt at cleaning up duplicate tracks that have different target values with the same artists.
## None of the songs have them, so it's perfectly clean.

print(data.columns)

print(data.nunique(axis=0))

sns.countplot(x=data['target'])
plt.show()

sns.heatmap(data.corr(), annot=True)
plt.show()

# Preprocessing
x = data.drop('target', axis=1)
y = data['target']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x = pd.DataFrame(data=x_scaled, columns=x.columns)
print(x.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
model = SVC()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
cm = confusion_matrix(y_test, predictions)
print(cm)
print(f"True Positives: {cm[0, 0]}")
print(f"\nTrue Negatives: {cm[0, 1]}")
print(f"\nFalse Positives: {cm[1, 0]}")
print(f"\nFalse Negatives: {cm[1, 1]}")

print(classification_report(y_test, predictions))

# Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True,verbose=3)
grid.fit(x_train, y_train)
print(grid.best_params_)

grid_predictions = grid.predict(x_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
