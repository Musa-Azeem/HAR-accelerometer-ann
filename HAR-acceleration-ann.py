import pandas as pd
import os

data_dir = 'smoking_data'
filename = 'smoking_input.csv'
df = pd.read_csv(os.path.join(data_dir, filename))
print(df)