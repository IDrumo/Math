import numpy as np
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import warnings

# Подавление предупреждений о сходимости
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Загрузка данных из Kaggle
path = kagglehub.dataset_download("hojjatk/mnist-dataset")
train_data = pd.read_csv(f"{path}/mnist_train.csv")

# Подготовка данных
X = train_data.drop('label', axis=1).values
y = train_data['label'].values

# Нормализация данных (важно для нейросетей)
X = X / 255.0

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


class WeightLogger:
    def __init__(self):
        self.weights_history = []

    def __call__(self, nn, epoch):
        # Получаем веса первого слоя
        weights_first_layer = nn.coefs_[0]
        self.weights_history.append(weights_first_layer.copy())

        # Выводим статистику по весам
        print(f"\nEpoch {epoch + 1}")
        print(f"Min weight: {weights_first_layer.min():.4f}")
        print(f"Max weight: {weights_first_layer.max():.4f}")
        print(f"Mean weight: {weights_first_layer.mean():.4f}")


# Создание и конфигурация модели
mlp = MLPClassifier(
    hidden_layer_sizes=(64,64),
    solver='adam',
    max_iter=10,
    random_state=1,
    verbose=True,
    n_iter_no_change=1000,
    early_stopping=False,
    tol=0
)

# Инициализация логгера
weight_logger = WeightLogger()

# Модифицированный цикл обучения для вывода весов
for epoch in range(mlp.max_iter):
    mlp.partial_fit(X_train, y_train, classes=np.unique(y))
    weight_logger(mlp, epoch)

    # Ручная проверка остановки
    if epoch > 0 and mlp._optimizer == 'adam':
        if np.allclose(mlp.loss_curve_[-2], mlp.loss_curve_[-1], rtol=1e-4):
            break

# Прогнозирование на тестовой выборке
y_pred = mlp.predict(X_test)

# Оценка модели
print("\n" + "=" * 40)
print("Classification Report:")
print("=" * 40)
print(classification_report(y_test, y_pred))

print("=" * 40)
print("Confusion Matrix:")
print("=" * 40)
print(confusion_matrix(y_test, y_pred))
print("=" * 40)