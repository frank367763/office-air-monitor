import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Чтение данных
base_dir = Path(__file__).resolve().parent
file_name = base_dir.parent / 'Мониторинг_офиса.txt'
df = pd.read_csv(file_name)

if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Вывод общей информации о данных
print("=== 1. Общая информация о данных ===")
print(f"Количество строк: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}")
print("\nТипы данных столбцов:")
print(df.dtypes)
print("\nДетальная сводка:")
df.info()

# 3. Проверка пропущенных значений
print("\n=== 2. Пропущенные значения ===")
missing_values = df.isna().sum()
print(missing_values)

# Гистограммы не показывают пустые ячейки — выводим долю пропусков отдельно.
missing_percent = df.isna().mean() * 100
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(missing_percent.index, missing_percent.values, color='salmon')
ax.bar_label(bars, labels=[f'{value:.1f}%' for value in missing_percent.values])
ax.set(title='Доля пропусков в исходных данных', ylabel='Пропуски, %', ylim=(0, 100))
plt.tight_layout()
plt.show()

# 4. Проверка и удаление дубликатов
duplicates_count = df.duplicated().sum()
print(f"\n=== 3. Проверка дубликатов ===")
print(f"Найдено дубликатов: {duplicates_count}")

# Показываем, сколько уникальных записей и лишних копий содержится в файле.
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(['Уникальные записи', 'Лишние копии'],
              [len(df) - duplicates_count, duplicates_count], color=['steelblue', 'salmon'])
ax.bar_label(bars)
ax.set(title='Полные дубликаты в исходных данных', ylabel='Количество строк')
plt.tight_layout()
plt.show()

if duplicates_count > 0:
    df = df.drop_duplicates()
    print(f"Дубликаты удалены. Итоговое количество строк: {len(df)}")
else:
    print("Дубликаты отсутствуют.")





def load_teds(filepath: str) -> dict:
    """Загружает файл TEDS и возвращает его содержимое в виде словаря Python."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


print("\n=== 4. Конфигурационный файл TEDS ===")
try:
    teds_data = load_teds(base_dir / 'teds_config.json')
    print("Конфигурация TEDS успешно загружена:")
    print(json.dumps(teds_data, indent=2, ensure_ascii=False))
except FileNotFoundError:
    print("Файл teds_config.json не найден. Создайте его рядом со скриптом.")





print("\n=== 5. Описательная статистика ===")
# Выбираем числовые столбцы: температура, влажность, CO2 и освещённость.
numeric_features = df.select_dtypes(include=['number']).columns.tolist()

stats_table = df[numeric_features].describe().rename(index={
    'count': 'Количество наблюдений',
    'mean': 'Среднее значение',
    'std': 'Стандартное отклонение',
    'min': 'Минимум',
    '25%': 'Первый квартиль (25%)',
    '50%': 'Медиана (50%)',
    '75%': 'Третий квартиль (75%)',
    'max': 'Максимум'
})

print(stats_table)




sns.set_theme(style="whitegrid")

# 1. Гистограммы распределения с наложенной кривой плотности (KDE)
fig, axes = plt.subplots(1, len(numeric_features), figsize=(15, 4))
if len(numeric_features) == 1:
    axes = [axes]
for ax, col in zip(axes, numeric_features):
    sns.histplot(df[col], kde=True, ax=ax, color='royalblue', bins=15)
    ax.set_title(f'Распределение: {col}')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Плотность / Частота')
plt.tight_layout()
plt.show()

# 2. Ящики для выявления выбросов
fig, axes = plt.subplots(1, len(numeric_features), figsize=(12, 4))
if len(numeric_features) == 1:
    axes = [axes]
for ax, col in zip(axes, numeric_features):
    sns.boxplot(y=df[col], ax=ax, color='mediumseagreen')
    ax.set_title(f'Boxplot: {col}')
plt.tight_layout()
plt.show()

# 3. Диаграммы рассеяния для пар признаков с цветовой кодировкой по классам
if 'label' in df.columns:
    pair_plot = sns.pairplot(
        data=df,
        vars=numeric_features,
        hue='label',
        palette='tab10',
        diag_kind='kde'
    )
    pair_plot.figure.subplots_adjust(top=0.92)
    pair_plot.figure.suptitle('Диаграммы рассеяния по классам')
    plt.show()

# 4. Графики временных рядов
if 'timestamp' in df.columns:
    fig, axes = plt.subplots(len(numeric_features), 1, figsize=(12, 3 * len(numeric_features)), sharex=True)
    if len(numeric_features) == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_features):
        sns.lineplot(
            data=df,
            x='timestamp',
            y=col,
            hue='label' if 'label' in df.columns else None,
            ax=ax,
            marker='o',
            palette='tab10'
        )
        ax.set_title(f'Временной ряд: {col}')
        ax.set_ylabel(col)
    plt.xlabel('Время')
    plt.tight_layout()
    plt.show()



# Выявление выбросов методом IQR

def find_outliers(data: pd.DataFrame, method: str = 'iqr') -> pd.Series:
    """
    Определяет выбросы для числовых признаков методом IQR.
    Возвращает булеву Series, где True означает выброс хотя бы по одному признаку.
    """
    if method != 'iqr':
        raise ValueError(f"Метод '{method}' не поддерживается. Используйте 'iqr'.")

    num_data = data.select_dtypes(include=['number'])

    # Расчет квартилей и межквартильного размаха
    q1 = num_data.quantile(0.25)
    q3 = num_data.quantile(0.75)
    iqr = q3 - q1

    # Границы допустимых значений: [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Маска значений, выходящих за допустимые границы
    outliers_mask = (num_data < lower_bound) | (num_data > upper_bound)

    # Определение строк, содержащих выброс хотя бы по одному признаку
    row_is_outlier = outliers_mask.any(axis=1)

    print(f"\n=== 6. Поиск выбросов (метод {method.upper()}) ===")
    print(f"Всего строк с выбросами: {row_is_outlier.sum()} из {len(data)}")
    print("\nКоличество выбросов по признакам:")
    print(outliers_mask.sum())

    return row_is_outlier


# Вызов функции поиска выбросов
df['is_outlier'] = find_outliers(df, method='iqr')

# Вывод найденных строк с аномалиями
if df['is_outlier'].any():
    print("\nСтроки, распознанные как выбросы:")
    print(df[df['is_outlier']])
else:
    print("\nВыбросов по методу IQR не обнаружено.")
