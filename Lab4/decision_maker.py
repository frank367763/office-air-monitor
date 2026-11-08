import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime
from math import isfinite


#-----------------------------
# 7.3.1. Загрузка модели и параметров масштабирования
#-----------------------------

def load_model(model_path, scaler_path):
    model = joblib.load(model_path)  # Готовая модель из третьей лабы — заново не обучаем.

    with open(scaler_path, 'r', encoding='utf-8') as file:
        scaler_params = json.load(file)  # Сохранённые минимумы и максимумы из второй лабы.

    # Собираем две строки: минимумы всех признаков и их максимумы.
    limits = pd.DataFrame({
        name: [values['min'], values['max']]
        for name, values in scaler_params.items()
    })
    limits = limits[list(model.feature_names_in_)]  # Тот же порядок, что при обучении.

    # Восстанавливаем прежнее масштабирование, чтобы модель получала привычные числа.
    scaler = MinMaxScaler()
    scaler.fit(limits)

    return model, scaler  # Возвращаем два результата: модель и преобразователь чисел.



#-----------------------------
# 7.3.2. Загрузка рекомендаций для каждого класса
#-----------------------------

def load_rules(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        rules = json.load(file)  # JSON превращается в словарь: класс → статус и совет.

    return rules



#-----------------------------
# 7.3.3. Подготовка данных и получение ответа модели
#-----------------------------

# Класс объединяет модель, настройки, рекомендации и журнал решений.
class DecisionMaker:
    # Выполняется при создании maker: загружает всё необходимое для работы.
    def __init__(self, model_path, scaler_path, rules_path, teds_path):
        self.load_model(model_path, scaler_path)
        self.load_rules(rules_path)
        self.load_teds(teds_path)

        self.log = []  # Пустой журнал; позже будем добавлять принятые решения.

        #7.3.6----------------------------------------------
        self.temperature_history = []  # Будем хранить две предыдущие температуры.
        self.previous_co2 = None  # Пока предыдущего измерения CO₂ нет.
        #--------------------------------------------------------------

    # Сохраняем оба результата готовой функции внутри объекта.
    def load_model(self, model_path, scaler_path):
        self.model, self.scaler = load_model(model_path, scaler_path)


    # Правила будут доступны другим методам через self.rules.
    def load_rules(self, rules_path):
        self.rules = load_rules(rules_path)

    # TEDS хранит описание данных и список признаков для модели.
    def load_teds(self, teds_path):
        with open(teds_path, 'r', encoding='utf-8') as file:
            self.teds = json.load(file)


    #7.3.6 доп+ ------------------------------------------------------
    # Дополняем показания и приводим числа к масштабу обучения.
    def preprocess(self, raw_data):
        raw_data = raw_data.copy()  # Не меняем исходный словарь пользователя.

        # Проверяем основные показания до расчёта среднего и изменения CO₂.
        for name in ['temperature', 'humidity', 'co2']:
            if name not in raw_data:
                raise ValueError(f'Не хватает показания: {name}.')
            try:
                raw_data[name] = float(raw_data[name])
            except (TypeError, ValueError):
                raise ValueError(f'{name}: нужно число.') from None

        # Если средняя температура уже есть в данных, оставляем готовую.
        if 'temp_smooth' not in raw_data:
            # К двум предыдущим температурам добавляем текущую.
            temperatures = self.temperature_history + [raw_data['temperature']]

            # Сумму делим на количество: получаем среднюю температуру.
            raw_data['temp_smooth'] = round(
                sum(temperatures) / len(temperatures), 2
            )

        # Если изменение CO₂ не передали, вычисляем сами.
        if 'co2_diff' not in raw_data:
            if self.previous_co2 is None:
                raw_data['co2_diff'] = 0  # Первое измерение не с чем сравнивать.
            else:
                raw_data['co2_diff'] = round(
                    raw_data['co2'] - self.previous_co2, 1
                )

        feature_names = self.teds['feature_fields']  # Порядок признаков модели.
        if feature_names != list(self.model.feature_names_in_):
            raise ValueError('Список признаков в TEDS не совпадает с признаками модели.')

        # Числовой формат ещё не означает, что показания подходят модели.
        for index, name in enumerate(feature_names):
            try:
                value = float(raw_data[name])
            except (TypeError, ValueError):
                raise ValueError(f'{name}: нужно число.') from None
            minimum = self.scaler.data_min_[index]
            maximum = self.scaler.data_max_[index]
            if not isfinite(value):
                raise ValueError(f'{name}: нужно конечное число, без NaN и бесконечности.')
            # Допуск нужен из-за погрешности обратного масштабирования тестовой таблицы.
            if value < minimum - 1e-8 or value > maximum + 1e-8:
                raise ValueError(f'{name}: значение вне диапазона {minimum:g}–{maximum:g}.')
            raw_data[name] = value

        data = pd.DataFrame([raw_data], columns=feature_names)

        scaled_data = self.scaler.transform(data)  # Применяем прежнее масштабирование.
        return pd.DataFrame(scaled_data, columns=feature_names)
    #доп--------------------------------------------------------------------------

    # Определяем состояние воздуха по подготовленным показаниям.
    def predict(self, features):
        prediction = self.model.predict(features)  # KNN возвращает ответ для каждой строки.
        return prediction[0]  # У нас одна строка: из ['comfort'] берём 'comfort'.

    # Находим статус и рекомендацию для предсказанного класса.
    def interpret(self, class_label):
        # Проверяем, есть ли правило для такого класса.
        if class_label in self.rules['classes']:
            return self.rules['classes'][class_label]
        # Если правило отсутствует, используем запасную рекомендацию.
        return {
            'status': 'НЕИЗВЕСТНО',
            'action': self.rules['default_action'],
            'severity': 'неизвестная',
            'color': 'серый'
        }

    # Добавляем одно решение в журнал и сохраняем его в файл.
    def log_decision(self, record):
        self.log.append(record)  # append добавляет запись в конец списка.
        self.save_log()

    # Сохраняем журнал текущего запуска в CSV.
    def save_log(self, filename='decisions_log.csv'):
        log_path = Path(__file__).resolve().parent / filename

        # Одна запись из списка становится одной строкой таблицы.
        log_table = pd.DataFrame(self.log)

        # index=False — не добавляем лишний столбец с номерами строк.
        log_table.to_csv(log_path, index=False, encoding='utf-8-sig')

    # Позволяем получить накопленные записи для проверки и статистики.
    def get_log(self):
        return self.log



    # Полный цикл: показания → класс → рекомендация → запись в журнал.
    def decide(self, raw_data, timestamp=None):
        # Если время измерения не передали, берём текущее.
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        features = self.preprocess(raw_data)  # Подготавливаем числа для модели.
        class_label = self.predict(features)  # Получаем состояние воздуха.
        interpretation = self.interpret(class_label)  # Находим правило для класса.

        # Для вывода возвращаем ровно те признаки, которые получила модель, в °C, %, ppm.
        feature_values = dict(zip(
            features.columns,
            self.scaler.inverse_transform(features)[0].tolist()
        ))
        first_manual_input = (
            'temp_smooth' not in raw_data and 'co2_diff' not in raw_data
            and not self.temperature_history and self.previous_co2 is None
        )

        # Собираем исходные данные и ответ системы в одну запись.
        result = {
            'timestamp': timestamp,
            'raw_data': raw_data,
            'features': feature_values,  # Пять фактически использованных признаков.
            'first_manual_input': first_manual_input,
            'class': class_label,
            'status': interpretation['status'],
            'action': interpretation['action'],
            'severity': interpretation['severity'],
            'color': interpretation['color']
        }

        self.log_decision(result)  # Сохраняем запись в журнал и CSV.

        # Запоминаем показания, если дополнительные признаки вычисляли сами.
        if 'temp_smooth' not in raw_data or 'co2_diff' not in raw_data:
            temperatures = self.temperature_history + [float(raw_data['temperature'])]
            self.temperature_history = temperatures[-2:]  # Последние две температуры.
            self.previous_co2 = float(raw_data['co2'])  # Прошлое значение для следующего ввода.

        return result

    # Показываем одно решение в понятном для пользователя виде.
    def display_decision(self, result):
        print('\n' + '=' * 50)  # Разделяем результаты разных измерений.
        print('Время:', result['timestamp'])
        print('\nВведённые показания:')
        print(f"  Температура: {float(result['raw_data']['temperature']):.2f} °C")
        print(f"  Влажность: {float(result['raw_data']['humidity']):.2f} %")
        print(f"  CO₂: {float(result['raw_data']['co2']):.2f} ppm")

        # В CSV признаки уже готовы; при ручном вводе они рассчитаны по истории.
        if 'temp_smooth' in result['raw_data'] and 'co2_diff' in result['raw_data']:
            print('\nДополнительные признаки из данных:')
        else:
            print('\nРассчитанные признаки:')
        print(f"  Средняя температура (temp_smooth): {result['features']['temp_smooth']:.2f} °C")
        print(f"  Изменение CO₂ (co2_diff): {result['features']['co2_diff']:+.2f} ppm")
        if result['first_manual_input']:
            print('  Первое измерение: средняя равна текущей температуре, изменение CO₂ равно 0.')

        print('\nРешение:')
        print('Класс:', result['class'])
        print('Статус:', result['status'])
        print('Рекомендация:', result['action'])
        print('Серьёзность:', result['severity'])
        print('=' * 50)
    

