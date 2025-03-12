import numpy as np
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Загрузка данных из Kaggle
path = kagglehub.dataset_download("hojjatk/mnist-dataset")
train_data = pd.read_csv(f"{path}/mnist_train.csv")

# Подготовка данных
X = train_data.drop('label', axis=1).values
y = train_data['label'].values

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Создание и обучение модели
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=1)
mlp.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred = mlp.predict(X_test)

# Оценка модели
print("\n" + "="*40)
print("Classification Report:")
print("="*40)
print(classification_report(y_test, y_pred))

print("="*40)
print("Confusion Matrix:")
print("="*40)
print(confusion_matrix(y_test, y_pred))
print("="*40)