from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression
from data_preprocessing import get_scaler_data

X_train_scaled, X_test_scaled, target_train, target_test = get_scaler_data()

# Создаем объект модели линейной регрессии
model = LinearRegression()

# Обучаем модель на нормализованных данных
model.fit(X_train_scaled, target_train)

# Предсказываем значения на тестовой выборке
target_pred = model.predict(X_test_scaled)

print(f'\nMSE: {MSE(target_pred, target_test).round(2)}')
