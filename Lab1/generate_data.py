import json
import random
from datetime import datetime, timedelta
from pathlib import Path

base_dir = Path(__file__).resolve().parent
output_paths = [
    base_dir / 'Мониторинг_офиса.txt',
    base_dir.parent / 'Мониторинг_офиса.txt'
]

start_time = datetime(2025, 3, 1, 8, 0, 0)
total_rows = 3000
rows_per_class = total_rows // 3

# Фиксируем генератор ДО создания значений, чтобы набор воспроизводился.
random.seed(42)

rows = []

for i in range(total_rows):
    current_time = start_time + timedelta(minutes=i)
    time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')

    # Идеальный баланс: по 1000 измерений на каждый класс.
    if i < rows_per_class:
        temp = round(random.uniform(20.5, 22.8), 1)
        hum = round(random.uniform(43.0, 52.0), 1)
        co2 = random.randint(420, 680)
        label = 'comfort'
    elif i < 2 * rows_per_class:
        temp = round(random.uniform(23.5, 26.2), 1)
        hum = round(random.uniform(55.0, 67.0), 1)
        co2 = random.randint(950, 1450)
        label = 'stuffy'
    else:
        temp = round(random.uniform(27.0, 31.0), 1)
        hum = round(random.uniform(70.0, 82.0), 1)
        co2 = random.randint(1900, 2700)
        label = 'critical'

    # Признак с частыми сбоями (> 50% пропусков)
    light = round(random.uniform(300.0, 500.0), 1)

    rows.append([time_str, temp, hum, co2, light, label])

#-----------------------------
# Явные сбои датчиков: 90 разных строк, по 30 в каждом классе
#-----------------------------

fault_indices = []
for class_number in range(3):
    start = class_number * rows_per_class
    fault_indices.extend(random.sample(range(start, start + rows_per_class), 30))

fault_set = set(fault_indices)
fault_records = []
field_names = ['timestamp', 'temperature', 'humidity', 'co2', 'light', 'label']
bad_values = {1: [2.0, 82.5], 2: [-15.0, 120.0], 3: [8800, 9200]}
for number, index in enumerate(fault_indices):
    column = 1 + number % 3
    bad_value = bad_values[column][(number // 3) % 2]
    fault_records.append({'timestamp': rows[index][0], 'field': field_names[column],
                          'original_value': rows[index][column], 'fault_value': bad_value})
    rows[index][column] = bad_value

#-----------------------------
# Пропуски: меньше 20% в основных датчиках и больше 50% в light
#-----------------------------

# Часть пропусков идёт подряд: такие серии заметны на временном графике.
# Сбои защищаем от пропусков, чтобы искусственные аномалии не исчезли.
missing_rates = {1: 0.15, 2: 0.17, 3: 0.14, 4: 0.60}
available = [i for i in range(total_rows) if i not in fault_set]
for column, rate in missing_rates.items():
    count = int(total_rows * rate)
    block_start = 80 + 120 * column
    block = {i for i in range(block_start, block_start + 35) if i not in fault_set}
    scattered = random.sample([i for i in available if i not in block], count - len(block))
    for index in block | set(scattered):
        rows[index][column] = ''

#-----------------------------
# Полные дубликаты: 90 строк (3%), по 30 в каждом классе
#-----------------------------

for class_number in range(3):
    start = class_number * rows_per_class
    # Не заменяем и не копируем сбои; метка класса при копировании сохраняется.
    candidates = [i for i in range(start + 1, start + rows_per_class)
                  if i not in fault_set and i - 1 not in fault_set]
    for index in sorted(random.sample(candidates, 30)):
        rows[index] = list(rows[index - 1])

# Запись
header = "timestamp,temperature,humidity,co2,light,label\n"
content = header + "\n".join(
    f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}" for r in rows
)

for path in output_paths:
    path.write_text(content, encoding='utf-8')

# Фиксируем фактические доли: копирование строк немного изменяет число пропусков.
missing_counts = {field_names[j]: sum(row[j] == '' for row in rows) for j in range(1, 5)}
summary = {
    'description': 'Синтетический учебный набор с искусственно добавленными дефектами',
    'seed': 42, 'rows': len(rows),
    'duplicates': len(rows) - len({tuple(row) for row in rows}),
    'missing_counts': missing_counts,
    'missing_percent': {name: round(count / len(rows) * 100, 2) for name, count in missing_counts.items()},
    'injected_faults': len(fault_records), 'fault_records': fault_records
}
(base_dir / 'generation_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"Создано {len(rows)} строк; дубликатов: {summary['duplicates']}; сбоев: {len(fault_records)}.")
print('Пропуски (%):', summary['missing_percent'])
print('После обновления данных заново выполните ЛР1 и ЛР2.')
