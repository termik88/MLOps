import pickle


# Состояние
class State:
    def __init__(self):
        self.X_train_scaled = None,
        self.X_test_scaled = None,
        self.target_train = None,
        self.target_test = None


# Функция сохранения состояния в файл
def save_state(state):
    with open('state.pkl', 'wb') as file:
        pickle.dump(state, file)


# Читаем объект из файла
def load_state():
    with open('state.pkl', 'rb') as file:
        return pickle.load(file)
