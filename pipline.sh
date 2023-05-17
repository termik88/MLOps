#!/bin/bash

train_dir="train"
test_dir="test"
target_dir="target"
scaler_dir="scaler"
predict_dir="predict"

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

# Проверяем наличие папки target
if [ -d "$target_dir" ]; then
  # Если папка target существует, удаляем ее со всем содержимым
  rm -r "$target_dir"
fi

# Проверяем наличие папки Scaler
if [ -d "$scaler_dir" ]; then
  # Если папка Scaler существует, удаляем ее со всем содержимым
  rm -r "$scaler_dir"
fi

# Проверяем наличие папки Predict
if [ -d "$predict_dir" ]; then
  # Если папка Predict существует, удаляем ее со всем содержимым
  rm -r "$predict_dir"
fi

# Создаем папки Train, Test, Scaler, Predict
mkdir "$train_dir"
mkdir "$test_dir"
mkdir "$target_dir"
mkdir "$scaler_dir"
mkdir "$predict_dir"

# Запуск файлов по порядку
echo "Запуск libraries.py..."
python libraries.py

echo "Запуск data_creation.py..."
python data_creation.py

echo "Запуск data_preprocessing.py..."
python data_preprocessing.py

echo "Запуск model_preparation.py..."
python model_preparation.py

echo "Запуск model_testing.py..."
python model_testing.py