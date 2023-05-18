from state import load_state
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression


def test_mse():
    # Читаем объект из файла
    state = load_state()

    # Создаем объект модели линейной регрессии
    model = LinearRegression()

    # Обучаем модель на нормализованных данных
    model.fit(state.X_train_scaled, state.target_train)

    # Предсказываем значения на тестовой выборке
    target_pred = model.predict(state.X_test_scaled)

    mse = MSE(target_pred, state.target_test).round(2)
    expected_mse = 615.55

    assert mse == expected_mse
