import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import json
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


base_dir = Path(__file__).resolve().parent

file_path = base_dir.parent / 'Мониторинг_офиса.txt'
if not file_path.exists():
    file_path = base_dir / 'Мониторинг_офиса.txt'

df = pd.read_csv(file_path)

print("--- 1. Размер датасета ---")
print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")

print("\n--- 2. Типы данных колонок ---")
print(df.dtypes)

print("\n--- 3. Первые 3 строки таблицы ---")
print(df.head(3))



df_raw = df.copy() #Сохраняем датасет для сравнения

# ЛР2 читает исходный файл заново: удаление повторов в ЛР1 не меняет этот файл.
# Убираем дубликаты до медиан, скользящего среднего и разбиения на выборки.
duplicates_count = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
df_raw = df.copy() # Сравниваем пропуски на одних и тех же уникальных строках.
print(f'\nУдалено полных дубликатов: {duplicates_count}; осталось строк: {len(df)}')

#-----------------------------
# Анализ пропусков и удаление столбцов (> 50% пропусков)
#-----------------------------

missing_share = df.isna().mean() * 100 #Процент пропусков

print("\n--- 4. Доля пропусков в каждом столбце ---")
print(missing_share.round(2))

cols_to_drop = missing_share[missing_share > 50].index.tolist() #Столбцы с > 50% пропусков
print(f"\nСтолбцы подлежащие удалению (> 50% пропусков): {cols_to_drop}")

if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"\nУдаление выполнено. Текущие столбцы датасета: {df.columns.tolist()}")


#-----------------------------
# Заполнение пропусков медианой (< 20%)
#-----------------------------

numeric_cols = df.select_dtypes(include=['number']).columns

print("\n--- 5. Заполнение пропусков медианой для числовых колонок (< 20% пропусков) ---")
for col in numeric_cols:
    share = missing_share[col]
    if 0 < share < 20:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val) #Метод fillna() заменяет пропуски на медиану
        print(f"Колонка '{col}': заполнено пропусков медианой ({median_val:.2f})")


print("\nКоличество пропусков после обработки:")
print(df.isna().sum())

fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(12, 3 * len(numeric_cols)))

for i, col in enumerate(numeric_cols):
    # Общие границы столбиков и одинаковые оси делают сравнение честным.
    bins = np.histogram_bin_edges(df_raw[col].dropna(), bins=30)
    filled_mask = df_raw[col].isna()
    # График слева: данные ДО обработки (из сохранённого df_raw)
    axes[i, 0].hist(df_raw[col].dropna(), bins=bins, color='steelblue')
    axes[i, 0].set_title(f'{col}: ДО; пропусков {filled_mask.sum()}')
    axes[i, 0].set_ylabel('Количество наблюдений')

    # График справа: данные ПОСЛЕ заполнения медианой (из текущего df)
    # Оранжевым выделяем заполненные значения, синим — исходные.
    axes[i, 1].hist([df.loc[~filled_mask, col], df.loc[filled_mask, col]],
                    bins=bins, stacked=True, color=['steelblue', 'darkorange'],
                    label=['Исходные значения', 'Заполнено медианой'])
    axes[i, 1].set_title(f'{col}: ПОСЛЕ; пропусков {df[col].isna().sum()}')
    axes[i, 1].set_ylabel('Количество наблюдений')
    axes[i, 1].legend()
    ymax = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
    for ax in axes[i]:
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, ymax)

plt.tight_layout()
plt.show()


#-----------------------------
# Поиск и удаление выбросов (IQR)
#-----------------------------

def find_outliers(data: pd.DataFrame, method: str = 'iqr') -> pd.Series:
    """
    Находит строки с аномальными значениями (выбросами) для числовых признаков методом IQR.
    Возвращает булеву Series, где True означает выброс хотя бы по одному признаку.
    """
    if method != 'iqr':
        raise ValueError(f"Метод '{method}' не поддерживается. Используйте 'iqr'.")

    num_data = data.select_dtypes(include=['number']) #Числовые столбцы

    q1 = num_data.quantile(0.25) #Первый квартиль
    q3 = num_data.quantile(0.75) #Третий квартиль
    iqr = q3 - q1 #Межквартильный размах

    #Границы нормальных значений по формуле Тьюки: [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Двумерная булева маска значений, выходящих за допустимые границы
    outliers_mask = (num_data < lower_bound) | (num_data > upper_bound) #Побитовое сравнение для каждой ячейки (побитовое потому что у нас DataFrame)

    # Проверка по строкам: True, если хотя бы один признак в строке является выбросом
    row_is_outlier = outliers_mask.any(axis=1) #axis=1 означает проверку по строкам

    print(f"\n=== 6. Поиск выбросов методом IQR ===")
    print(f"Всего строк с аномалиями: {row_is_outlier.sum()} из {len(data)}")
    print("\nКоличество выбросов по признакам:")
    print(outliers_mask.sum())

    return row_is_outlier


# Вызов функции поиска выбросов
outlier_mask = find_outliers(df, method='iqr')

print(f"\nОбнаруженные строки со сбоем датчиков:")
print(df[outlier_mask])

df_clean = df[~outlier_mask].copy().reset_index(drop=True) #Удаляем строки с выбросами и сбрасываем индексы, ~ означает инвертирование булевой маски (True -> False, False -> True)

# На одинаковых осях показываем, какие экстремальные значения удалены.
fig, axes = plt.subplots(2, len(numeric_cols), figsize=(14, 7), squeeze=False)
for i, col in enumerate(numeric_cols):
    sns.boxplot(y=df[col], ax=axes[0, i], color='salmon')
    sns.boxplot(y=df_clean[col], ax=axes[1, i], color='mediumseagreen')
    axes[0, i].set_title(f'{col}: до удаления выбросов')
    axes[1, i].set_title(f'{col}: после удаления выбросов')
    axes[1, i].set_ylim(axes[0, i].get_ylim())
fig.suptitle(f'Удаление выбросов: исключено {int(outlier_mask.sum())} строк')
plt.tight_layout()
plt.show()

print(f"\nРазмер датасета до удаления выбросов: {df.shape[0]} строк | после удаления: {df_clean.shape[0]} строк")
print(f"Удалено {df.shape[0] - df_clean.shape[0]} строк с выбросами.")

df = df_clean #Обновляем df на очищенный датасет


#-----------------------------
# Создание новых признаков (feature engineering)
#-----------------------------

df['temp_smooth'] = df['temperature'].rolling(window=3, min_periods=1).mean().round(2) #Скользящее среднее температуры с окном 3 и округлением до 2 знаков

df['co2_diff'] = df['co2'].diff().fillna(0).round(1) #Разница CO2 между текущей и предыдущей строкой, пропуски заполняем 0 и округляем до 1 знака

print("\n--- 7. Первые 5 строк после добавления новых признаков ---")
print(df[['timestamp', 'temperature', 'temp_smooth', 'co2', 'co2_diff']].head(5))



#-----------------------------
# Масштабирование, разбиение и сохранение
#-----------------------------

feature_cols = ['temperature', 'humidity', 'co2', 'temp_smooth', 'co2_diff']

min_max_scaler = MinMaxScaler() # Инициализация MinMaxScaler
std_scaler = StandardScaler() # Инициализация StandardScaler

df_minmax = df.copy()
df_minmax[feature_cols] = min_max_scaler.fit_transform(df[feature_cols]) #Масштабирование в диапазон [0, 1]

df_std = df.copy()
df_std[feature_cols] = std_scaler.fit_transform(df[feature_cols]) #Масштабирование с Z-нормализацией (среднее=0, std=1)

# График сравнения исходных данных, MinMax и Standard
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.histplot(df['temperature'], kde=True, ax=axes[0], color='gray', bins=15)
axes[0].set_title('Исходная температура (°C)')
axes[0].set_xlabel('temperature')

sns.histplot(df_minmax['temperature'], kde=True, ax=axes[1], color='teal', bins=15)
axes[1].set_title('MinMax нормализация [0, 1]')
axes[1].set_xlabel('temperature')

sns.histplot(df_std['temperature'], kde=True, ax=axes[2], color='purple', bins=15)
axes[2].set_title('Стандартизация (StandardScaler)')
axes[2].set_xlabel('temperature')

plt.tight_layout()
plt.show()


# Разделение выборки на Train (60%), Val (20%) и Test (20%) со стратификацией
# Для обучения берём датасет после MinMax нормализации, чтобы модель видела данные в диапазоне [0, 1]
x = df_minmax.drop(columns=['label'])
y = df_minmax['label']

# Шаг 1: отделяем 60% на Train, 40% во временную выборку temp
X_train, X_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.4, random_state=42, stratify=y
) # Метод train_test_split() из sklearn разделяет данные на обучающую и временную выборку, test_size=0.4 означает, что 40% данных уйдёт во временную выборку, random_state=42 фиксирует генератор случайных чисел для воспроизводимости, stratify=y обеспечивает стратификацию по классам меток
# Короче возвращает 60% строк для обучения и 40% для временной выборки

# Шаг 2: оставшиеся 40% делим ровно пополам (по 20% от общего объёма)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
) # Делим ещё пополам, 20% на валидацию и 20% на тест, stratify=y_temp обеспечивает стратификацию по классам меток во временной выборкe


print("\n=== 8. Размеры сформированных выборок ===")
print(f"Обучающая (Train 60%):    {len(X_train)} строк")
print(f"Валидационная (Val 20%):  {len(X_val)} строк")
print(f"Тестовая (Test 20%):      {len(X_test)} строк")

# Проверка баланса классов
train_balance = (y_train.value_counts(normalize=True) * 100).round(1).to_dict() 
print(f"\nБаланс классов в Train (%): {train_balance}")

# 5. Размечаем выборки в общем датасете
df_processed = df_minmax.copy()
df_processed['split'] = 'train'
df_processed.loc[X_val.index, 'split'] = 'val'
df_processed.loc[X_test.index, 'split'] = 'test'

# 6. Сохранение артефактов на диск

# А. Параметры нормализации для продакшена
scaler_params = {
    col: {
        'min': float(min_max_scaler.data_min_[i]),
        'max': float(min_max_scaler.data_max_[i])
    }
    for i, col in enumerate(feature_cols)
}

params_path = base_dir / 'scaler_params.json'
with open(params_path, 'w', encoding='utf-8') as f:
    json.dump(scaler_params, f, ensure_ascii=False, indent=2)
print(f"\nПараметры масштабирования сохранены в: {params_path.name}")

# Б. Полный обработанный датасет для обучения моделей
processed_path = base_dir / 'sensor_data_processed.csv'
df_processed.to_csv(processed_path, index=False, encoding='utf-8')
print(f"Предобработанный датасет сохранен в: {processed_path.name}")

# В. Чистый тестовый срез (20%) для проверки готовой системы
test_df = df_minmax.loc[X_test.index].copy()
test_path = base_dir / 'sensor_data_test.csv'
test_df.to_csv(test_path, index=False, encoding='utf-8')
print(f"Тестовая выборка сохранена в: {test_path.name}")

