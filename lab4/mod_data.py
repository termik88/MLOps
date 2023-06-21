import numpy as np
import pandas as pd

df = pd.read_csv("dataset/df.csv", index_col=0)
df['y'] = df['y'].mean()
df.to_csv("dataset/df.csv")