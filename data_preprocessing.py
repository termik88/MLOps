from libraries import get_lib

pd, StandardScaler = get_lib()

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

X_train_scaled.to_csv('slacer/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('slacer/X_test_scaled.csv', index=False)
target_train.to_csv('target/target_train.csv', index=False)
target_test.to_csv('target/target_test.csv', index=False)
