import numpy as np
import pandas as pd

np.random.seed(1)
sampleSize = 1000

# рандомный дата сет
x = 20 * np.random.randn(sampleSize) + 100
y = 30 * np.random.randn(sampleSize) + 200
t = 300 - y + 5 * np.random.randn(sampleSize)
z = 5 * np.random.randn(sampleSize) + 20

# Шум
e = np.random.randn(sampleSize) * 0.05

# Целевая переменная через линейное уравнение
target = 2 * x + 8 * y + z + 25 * np.random.randn(sampleSize) + e


df = pd.DataFrame(
    list(zip(x, y, t, z, target)),
    columns=['x', 'y', 't', 'z', 'target']
)
df = df.round(0)

df[:700].to_csv('train/train.csv', index=False)
df[700:].to_csv('test/test.csv', index=False)
