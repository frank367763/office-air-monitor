import json  # Чтение конфигурации в формате JSON.
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
import time

import joblib
import shutil
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import pandas as pd


#-----------------------------
# Загрузка обработанных данных и конфигурации TEDS
#-----------------------------

# Получаем полный путь к папке скрипта, независимо от папки запуска.
base_dir = Path(__file__).resolve().parent

project_dir = base_dir.parent


data = pd.read_csv(
    project_dir / 'Lab2' / 'sensor_data_processed.csv'
)


with open(
    base_dir / 'teds_config.json',
    'r',
    encoding='utf-8'
) as file:
    teds = json.load(file)  # Преобразуем JSON в словарь Python.

#-----------------------------
# Первичный просмотр таблицы и её описания
#-----------------------------

print('Размер таблицы:', data.shape)  # (Количество строк, количество столбцов).

print('\nСтолбцы таблицы:')
print(data.columns.tolist())  # Названия столбцов в виде списка.

print('\nПервые пять строк:')
print(data.head())  # Первые 5 строк для проверки содержимого.

print('\nПризнаки согласно TEDS:')
print(teds['feature_fields'])

print('\nЦелевая переменная согласно TEDS:')
print(teds['target_field'])

#-----------------------------
# Проверка пропусков, размеров выборок и баланса классов
#-----------------------------

print('\nКоличество пропусков:')
print(data.isna().sum())  # isna() отмечает пропуски, sum() считает их по столбцам.

print('\nРазмеры выборок:')
print(data['split'].value_counts())  # Сколько строк помечено как train, val и test.

print('\nКоличество объектов каждого класса:')
print(data['label'].value_counts())  # Сколько примеров каждого состояния офиса.

print('\nДоля каждого класса в процентах:')
# normalize=True возвращает доли; * 100 переводит их в проценты; round(2) округляет.
print(
    (data['label'].value_counts(normalize=True) * 100).round(2)
)

#-----------------------------
# Выделение признаков X и правильных ответов y
#-----------------------------

# Берём названия из TEDS, чтобы не перечислять столбцы повторно в коде.
feature_columns = teds['feature_fields']
target_column = teds['target_field']

# X — пять признаков: temperature, humidity, co2, temp_smooth и co2_diff.
# y — известное состояние: comfort, stuffy или critical.
# label, timestamp и split не включаем в X.
X = data[feature_columns]
y = data[target_column]

#-----------------------------
# Извлечение Train, Val и Test по разметке второй лабораторной
#-----------------------------

# == сравнивает значения: маска содержит True для нужных строк и False для остальных.
# Данные не делим заново — используем уже сохранённый столбец split.
train_mask = data['split'] == 'train'
val_mask = data['split'] == 'val'
test_mask = data['split'] == 'test'

# .loc[маска] оставляет строки с True. Одинаковая маска сохраняет связь X и y.
# Train — примеры для обучения модели.
X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

# Val — отдельные примеры для сравнения и выбора моделей.
X_val = X.loc[val_mask]
y_val = y.loc[val_mask]

# Test — итоговая проверка; не используем её для подбора настроек.
X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

#-----------------------------
# Проверка подготовленных данных перед обучением
#-----------------------------

# Например, (897, 5) — 897 строк с 5 признаками, (897,) — 897 ответов.
print('\nРазмеры признаков и ответов:')
print('Train:', X_train.shape, y_train.shape)
print('Val:', X_val.shape, y_val.shape)
print('Test:', X_test.shape, y_test.shape)

print('\nПервые обучающие примеры:')
print(X_train.head())

print('\nПравильные ответы для этих примеров:')
# Индексы ответов должны совпадать с индексами соответствующих строк X_train.
print(y_train.head())


#-----------------------------
# Создание и обучение KNN
#-----------------------------

# Пока берём 5 соседей; лучшее количество подберём следующим шагом.
knn = KNeighborsClassifier(n_neighbors=5)

# Передаём модели показания и соответствующие правильные ответы.
knn.fit(X_train, y_train)

print('\nKNN получила обучающие примеры.')
print('Количество соседей:', knn.n_neighbors)
print('Классы, известные модели:', knn.classes_)



#-----------------------------
# Подбор количества соседей KNN
#-----------------------------

knn_parameters = {
    'n_neighbors': [1, 3, 5, 7, 10, 15, 20]
}

# Пять разбиений с сохранением пропорций классов.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Организуем перебор настроек и сравнение по среднему F1 классов.
knn_search = GridSearchCV(
    estimator=knn,
    param_grid=knn_parameters,
    scoring='f1_macro',
    cv=cv
)

# Запускаем подбор только на обучающей выборке.
knn_search.fit(X_train, y_train)

print('\nЛучшие параметры KNN:')
print(knn_search.best_params_)

print('\nСредний F1 на кросс-валидации:')
print(round(knn_search.best_score_, 4))

# Получаем модель с выбранной настройкой, обученную на всём train.
best_knn = knn_search.best_estimator_

# Замеряем обучение KNN с выбранными параметрами.
start_time = time.perf_counter()
best_knn.fit(X_train, y_train)
knn_training_time = time.perf_counter() - start_time


#-----------------------------
# Проверка KNN на тестовой выборке
#-----------------------------

# Передаём только признаки: правильные ответы модель не получает.
y_pred_knn = best_knn.predict(X_test)

# Ставим рядом настоящие и предсказанные классы.
knn_comparison = pd.DataFrame({
    'Правильный класс': y_test,
    'Ответ KNN': y_pred_knn
})

print('\nПервые 10 результатов KNN:')
print(knn_comparison.head(10))


# Accuracy — доля совпадений предсказанных и правильных классов.
knn_accuracy = accuracy_score(y_test, y_pred_knn)

print('\nAccuracy KNN:')
print(round(knn_accuracy, 4))

print('\nМетрики по каждому классу:')
print(classification_report(y_test, y_pred_knn, zero_division=0))




#-----------------------------
# Создание и обучение дерева решений
#-----------------------------

# Пока ограничиваем глубину дерева тремя уровнями.
tree_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# Дерево подбирает проверки признаков по обучающим примерам.
tree_model.fit(X_train, y_train)

print('\nДерево решений обучено.')
print('Глубина дерева:', tree_model.get_depth())
print('Количество листьев:', tree_model.get_n_leaves())


#-----------------------------
# Подбор параметров дерева решений
#-----------------------------

# Проверяем глубину и минимальное число примеров для разделения узла.
tree_parameters = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10]
}

# Используем те же пять разбиений train и тот же критерий, что для KNN.
tree_search = GridSearchCV(
    estimator=tree_model,
    param_grid=tree_parameters,
    scoring='f1_macro',
    cv=cv
)

# Подбираем настройки только по обучающим данным.
tree_search.fit(X_train, y_train)

print('\nЛучшие параметры дерева:')
print(tree_search.best_params_)

print('\nСредний F1 дерева на кросс-валидации:')
print(round(tree_search.best_score_, 4))

# Выбранная модель уже обучена на всём train.
best_tree = tree_search.best_estimator_

# Замеряем обучение дерева с выбранными параметрами.
start_time = time.perf_counter()
best_tree.fit(X_train, y_train)
tree_training_time = time.perf_counter() - start_time


#-----------------------------
# Проверка выбранного дерева на тестовой выборке
#-----------------------------

# Передаём показания; получаем предсказанные состояния.
y_pred_tree = best_tree.predict(X_test)

# Сравниваем предсказания с правильными ответами.
tree_accuracy = accuracy_score(y_test, y_pred_tree)

print('\nAccuracy дерева:')
print(round(tree_accuracy, 4))

print('\nМетрики дерева по каждому классу:')
print(classification_report(y_test, y_pred_tree, zero_division=0))






#-----------------------------
# Создание и обучение нейросети MLP
#-----------------------------

# Один скрытый слой из 25 нейронов.
mlp_model = MLPClassifier(
    hidden_layer_sizes=(25,),
    max_iter=1000,
    random_state=42
)

# Нейросеть подбирает внутренние веса по показаниям и правильным ответам.
mlp_model.fit(X_train, y_train)

print('\nНейросеть обучена.')
print('Количество выполненных итераций:', mlp_model.n_iter_)


#-----------------------------
# Подбор параметров нейросети MLP
#-----------------------------

mlp_parameters = {
    'hidden_layer_sizes': [(25,), (50,), (25, 25), (50, 25)],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'alpha': [0.0001, 0.001, 0.01]
}

# Проверяем все сочетания настроек на тех же разбиениях train.
mlp_search = GridSearchCV(
    estimator=mlp_model,
    param_grid=mlp_parameters,
    scoring='f1_macro',
    cv=cv
)

print('\nНачинаем подбор параметров MLP...')


mlp_search.fit(X_train, y_train)

print('\nЛучшие параметры MLP:')
print(mlp_search.best_params_)

print('\nСредний F1 MLP на кросс-валидации:')
print(round(mlp_search.best_score_, 4))

# Выбранная нейросеть уже обучена на всём train.
best_mlp = mlp_search.best_estimator_

# Замеряем обучение MLP с выбранными параметрами.
start_time = time.perf_counter()
best_mlp.fit(X_train, y_train)
mlp_training_time = time.perf_counter() - start_time


#-----------------------------
# Проверка выбранной нейросети на тестовой выборке
#-----------------------------

# Передаём показания; получаем предсказанные классы.
y_pred_mlp = best_mlp.predict(X_test)

# Сравниваем ответы нейросети с правильными ответами.
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)

print('\nAccuracy MLP:')
print(round(mlp_accuracy, 4))

print('\nМетрики MLP по каждому классу:')
print(classification_report(y_test, y_pred_mlp, zero_division=0))






#-----------------------------
# Сравнение качества трёх моделей
#-----------------------------

# Для каждой модели храним название, предсказания и время обучения.
model_predictions = [
    ('KNN', y_pred_knn, knn_training_time),
    ('Дерево', y_pred_tree, tree_training_time),
    ('MLP', y_pred_mlp, mlp_training_time)
]

# Сюда будем добавлять строку результатов для каждой модели.
comparison_rows = []

for model_name, predictions, training_time in model_predictions:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average='macro',
        zero_division=0
    )

    comparison_rows.append({
        'Модель': model_name,
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Время обучения, с': training_time
    })

# Превращаем список результатов в таблицу.
model_comparison = pd.DataFrame(comparison_rows)

print('\nСравнение моделей на тестовой выборке:')
print(model_comparison.round(4).to_string(index=False))





#-----------------------------
# Матрицы ошибок для трёх моделей
#-----------------------------

# Одинаковый порядок классов на всех графиках.
class_names = ['comfort', 'critical', 'stuffy']

for model_name, predictions, training_time in model_predictions:
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=class_names,
        cmap='Blues',
        colorbar=False,
        values_format='d'
    )

    plt.title(f'Матрица ошибок: {model_name}')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Настоящий класс')
    plt.tight_layout()

    plt.savefig(
        base_dir / f'confusion_matrix_{model_name}.png',
        dpi=150
    )

plt.show()



#-----------------------------
# Важность признаков дерева — пункт 6.3.6
#-----------------------------

# Дерево возвращает оценку важности для каждого входного признака.
feature_importance = pd.Series(
    best_tree.feature_importances_,
    index=feature_columns,
    name='Важность'
)

print('\nВажность признаков дерева:')
print(feature_importance.sort_values(ascending=False))

# Сортируем значения для удобного чтения графика.
plt.figure(figsize=(9, 5))
feature_importance.sort_values().plot.barh()
plt.title('Важность признаков дерева решений')
plt.xlabel('Важность')
plt.tight_layout()
plt.savefig(base_dir / 'feature_importance.png', dpi=150)
plt.show()

feature_importance.to_csv(
    base_dir / 'feature_importance.csv',
    encoding='utf-8-sig'
)

#-----------------------------
# Выбор модели по валидационной выборке
#-----------------------------

# При одинаковом F1 предпочитаем дерево как более понятную модель.
models = {
    'Дерево': best_tree,
    'KNN': best_knn,
    'MLP': best_mlp
}

validation_scores = {}

for model_name, model in models.items():
    validation_predictions = model.predict(X_val)

    validation_scores[model_name] = f1_score(
        y_val,
        validation_predictions,
        average='macro'
    )

print('\nF1 моделей на валидационной выборке:')
print(validation_scores)

# Находим название с наибольшим значением F1.
best_model_name = max(
    validation_scores,
    key=validation_scores.get
)
best_model = models[best_model_name]

print('\nВыбранная модель:', best_model_name)

#-----------------------------
# Сохранение результатов и модели для ЛР4
#-----------------------------

model_comparison.to_csv(
    base_dir / 'model_comparison.csv',
    index=False,
    encoding='utf-8-sig'
)

best_parameters = {
    'KNN': knn_search.best_params_,
    'Дерево': tree_search.best_params_,
    'MLP': mlp_search.best_params_
}

with open(base_dir / 'best_parameters.json', 'w', encoding='utf-8') as file:
    json.dump(best_parameters, file, ensure_ascii=False, indent=2)

# Сохраняем обученную модель: в ЛР4 её можно загрузить без повторного обучения.
joblib.dump(best_model, base_dir / 'best_model.joblib')

# Модели учились на данных, масштабированных в ЛР2:
# для новых показаний потребуются именно те же параметры.
shutil.copy2(
    project_dir / 'Lab2' / 'scaler_params.json',
    base_dir / 'scaler_params.json'
)

with open(base_dir / 'selected_model.json', 'w', encoding='utf-8') as file:
    json.dump(
        {
            'model': best_model_name,
            'features': feature_columns,
            'validation_f1': validation_scores[best_model_name]
        },
        file,
        ensure_ascii=False,
        indent=2
    )

print('\nРезультаты и модель сохранены в папке Lab3.')