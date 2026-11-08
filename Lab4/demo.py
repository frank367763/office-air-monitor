from pathlib import Path
from decision_maker import DecisionMaker  # Берём наш класс из соседнего файла.

import pandas as pd
import argparse

#7.3.6-------------------------------------------

# Ручной ввод: повторяем измерения, пока пользователь не напишет exit.
def interactive_mode(maker):
    print('\n--- Ручной ввод ---')
    print('Введите температуру °C, влажность % и CO₂ ppm.')
    print('Для выхода напишите exit вместо любого значения.')

    while True:
        raw_data = {}  # Новое измерение — новый словарь.

        for field in ['temperature', 'humidity', 'co2']:
            try:
                value = input(f'{field}: ').strip()  # Убираем пробелы по краям.
            except (EOFError, KeyboardInterrupt):
                print('\nРучной ввод завершён.')
                return

            if value.lower() == 'exit':
                return  # Завершаем ручной режим.

            try:
                raw_data[field] = float(value.replace(',', '.'))
            except ValueError:
                print('Нужно число. Начните ввод измерения заново.')
                break
        else:
            # Сюда попадаем, только если все три числа введены без break.
            try:
                result = maker.decide(raw_data)
            except ValueError as error:
                print('Измерение не обработано:', error)
            else:
                maker.display_decision(result)

#------------------------------
#7.3.4 
#------------------------------
# Запуск демонстрации работы системы.
def main(batch_only=False):
    base_dir = Path(__file__).resolve().parent

    # Загружаем модель и настройки один раз при запуске.
    maker = DecisionMaker(
        model_path=base_dir / 'best_model.joblib',
        scaler_path=base_dir / 'scaler_params.json',
        rules_path=base_dir / 'decision_rules.json',
        teds_path=base_dir / 'teds_config.json'
    )



    # Загружаем тестовую выборку из второй лабы.
    data = pd.read_csv(base_dir.parent / 'Lab2' / 'sensor_data_test.csv')
    feature_names = maker.teds['feature_fields']
    if data.empty:
        raise ValueError('Тестовая таблица пуста.')

    # Возвращаем исходные единицы: например, 0.429 → 25 °C.
    data[feature_names] = maker.scaler.inverse_transform(data[feature_names])

    print('\n--- Обработка тестовой выборки ---')
    print('Количество измерений:', len(data))


    #7.3.5 ------------------------------------------------
    correct = 0  # Счётчик правильных ответов.
    errors = []  # Список ошибок: настоящий класс и ответ модели.
    #--------------------------------------------------------------

    # Последовательно передаём каждое измерение нашему модулю.
    for index, row in data.iterrows():
        # Берём только признаки: правильный ответ label модели не передаём.
        raw_data = row[feature_names].to_dict()

        result = maker.decide(raw_data, row['timestamp'])

        #7.3.5 ------------------------------------------------
        # Правильный ответ используем только для проверки, после предсказания.
        true_class = row['label']
        predicted_class = result['class']

        if predicted_class == true_class:
            correct += 1  # То же самое, что correct = correct + 1.
        else:
            errors.append({
                'timestamp': row['timestamp'],
                'true': true_class,
                'predicted': predicted_class
            })
        #--------------------------------------------------------------

        # На экран выводим первые 5 решений; в журнал попадают все.
        if index < 5:
            maker.display_decision(result)

    print('\nВсего решений в журнале:', len(maker.get_log()))
    
    # Доля правильных ответов среди всех измерений.
    accuracy = correct / len(data) * 100

    print('\n--- Результаты проверки ---')
    print('Всего измерений:', len(data))
    print('Правильных решений:', correct)
    print('Ошибок:', len(errors))
    print(f'Точность: {accuracy:.2f}%')  # .2f — два знака после точки.

    if errors:
        # Считаем, сколько ошибок пришлось на каждый настоящий класс.
        error_table = pd.DataFrame(errors)
        print('\nОшибки по настоящим классам:')
        print(error_table['true'].value_counts())

        print('\nНастоящий класс и ошибочный ответ:')
        print(error_table.to_string(index=False))

    # После проверки таблицы переходим к ручному вводу.
    if not batch_only:
        interactive_mode(maker)
# Вызываем main только при непосредственном запуске этого файла.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Проверка модуля и ручной ввод показаний.')
    parser.add_argument('--batch', action='store_true', help='Только проверка таблицы, без ручного ввода.')
    args = parser.parse_args()
    try:
        main(batch_only=args.batch)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print('Не удалось выполнить запуск:', error)
        raise SystemExit(1)
