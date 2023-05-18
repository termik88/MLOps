import pandas as pd
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('train/train.csv', delimiter=',')
test_data = pd.read_csv('test/test.csv', delimiter=',')

# Отделяем целевую переменную (y) от признаков (X) в тренировочной выборке
X_train = train_data.drop('target', axis=1)
target_train = train_data['target']

# Отделяем целевую переменную (y) от признаков (X) в тестовой выборке
X_test = test_data.drop('target', axis=1)
target_test = test_data['target']

# Создаем объект стандартизации
scaler = StandardScaler()

# Применяем стандартизацию на тренировочной выборке
X_train_scaled = scaler.fit_transform(X_train)

# Применяем стандартизацию на тестовой выборке
X_test_scaled = scaler.transform(X_test)

def get_scaler_data():
    return X_train_scaled, X_test_scaled, target_train, target_test
