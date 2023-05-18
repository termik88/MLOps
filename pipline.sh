#!/bin/bash

train_dir="train"
test_dir="test"


# Проверяем наличие папки Train
if [ -d "$train_dir" ]; then
  # Если папка Train существует, удаляем ее со всем содержимым
  rm -r "$train_dir"
fi

# Проверяем наличие папки Test
if [ -d "$test_dir" ]; then
  # Если папка Test существует, удаляем ее со всем содержимым
  rm -r "$test_dir"
fi

# Создаем папки Train, Test, Scaler, Predict
mkdir "$train_dir"
mkdir "$test_dir"

# Запуск файлов по порядку
echo "Запуск data_creation.py..."
python3 data_creation.py

echo "Запуск data_preprocessing.py..."
python3 data_preprocessing.py

echo "Запуск model_preparation.py..."
python3 model_preparation.py

#echo "Запуск model_testing.py..."
#python3 model_testing.py