from state import State, save_state
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Создаем объект класса Stats - для хранения состояния модели
state = State()

train_data = pd.read_csv('train/train.csv', delimiter=',')
test_data = pd.read_csv('test/test.csv', delimiter=',')

# Отделяем целевую переменную (y) от признаков (X) в тренировочной выборке
X_train = train_data.drop('target', axis=1)
state.target_train = train_data['target']

# Отделяем целевую переменную (y) от признаков (X) в тестовой выборке
X_test = test_data.drop('target', axis=1)
state.target_test = test_data['target']

# Создаем объект стандартизации
scaler = StandardScaler()

# Применяем стандартизацию на тренировочной выборке
state.X_train_scaled = scaler.fit_transform(X_train)

# Применяем стандартизацию на тестовой выборке
state.X_test_scaled = scaler.transform(X_test)

# Сохраняем объект в файл с помощью pickle
save_state(state)
