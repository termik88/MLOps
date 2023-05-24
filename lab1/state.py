import pickle
import os


# Состояние
class State:
    def __init__(self):
        self.X_train_scaled = None,
        self.X_test_scaled = None,
        self.target_train = None,
        self.target_test = None


file_name = 'state.pkl'


# Получить путь до файлов
def get_path(current_directory, file_name):
    return os.path.join(current_directory, file_name)


# Функция сохранения состояния в файл
def save_state(current_directory, state):
    file_path = get_path(current_directory, file_name)
    with open(file_path, 'wb') as file:
        pickle.dump(state, file)


# Читаем объект из файла
def load_state(current_directory):
    file_path = get_path(current_directory, file_name)
    with open(file_path, 'rb') as file:
        return pickle.load(file)
