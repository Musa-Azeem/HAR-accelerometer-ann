import pandas as pd
import os
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

data_dir = 'smoking_data'
X_filename = 'smoking_input.csv'
Y_filename = 'smoking_targets.csv'

X = pd.read_csv(os.path.join(data_dir, X_filename), nrows=100, header=None)
y = pd.read_csv(os.path.join(data_dir, Y_filename), nrows=100, header=None)
X = X[X[299] != np.NaN]
print(X[299].loc[0])
# fig = px.line(X.iloc[0])
# fig.show(renderer='browser')

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)
# clf = MLPClassifier(10)
# clf.fit(X_train, y_train)
# print(clf.predict(X_test))
# clf.score(X_test, y_test)