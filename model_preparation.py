from libraries import get_lib

pd, LinearRegression, MSE = get_lib()

X_train_scaled = pd.read_csv('scaler/X_train_scaled.csv', delimiter=',')
X_test_scaled = pd.read_csv('scaler/X_test_scaled.csv', delimiter=',')
target_train = pd.read_csv('target/target_train.csv', delimiter=',')
target_test = pd.read_csv('target/target_test.csv', delimiter=',')

# Создаем объект модели линейной регрессии
model = LinearRegression()

# Обучаем модель на нормализованных данных
model.fit(X_train_scaled, target_train)

# Предсказываем значения на тестовой выборке
target_pred = model.predict(X_test_scaled)

print(f'\nMSE: {MSE(target_pred, target_test).round(2)}')

target_pred.to_csv('predict/target_pred.csv', index=False)
