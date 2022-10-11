import pandas as pd
import os
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

data_dir = 'smoking_data'
X_filename = 'smoking_input.csv'
Y_filename = 'smoking_targets.csv'
nrows = 1000

df = pd.read_csv(os.path.join(data_dir, X_filename), nrows=nrows, header=None)
df = pd.concat([df, 
               pd.read_csv(os.path.join(data_dir, Y_filename), nrows=nrows, header=None, names=['Label'])],
               axis=1)

# impute NaN values with mean of column
# imp_mean = SimpleImputer()
# imp_mean.fit(df)
# df = pd.DataFrame(imp_mean.transform(df))
df = df.dropna()#.reset_index(drop=True)

# fig = px.line(X.iloc[0])
# fig.show(renderer='browser')

X = df.drop(columns='Label')
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
clf = MLPClassifier(11)
clf.fit(X_train, y_train)
p = clf.predict(X_test)
score = clf.score(X_test, y_test)
print(score)
print(clf.coefs_)