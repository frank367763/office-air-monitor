# Office Air Monitor

По синтетическим показаниям датчиков определяет состояние воздуха в помещении. В проекте четыре шага: генерация данных, предобработка, сравнение классификаторов и проверка рекомендаций по результатам модели.

## Запуск

Из корня проекта в PowerShell:

```powershell
python -m pip install -r requirements.txt
$env:MPLBACKEND = 'Agg'
python Lab1/generate_data.py
python Lab2/lab2_preprocessing.py
python Lab3/lab3_training.py
Copy-Item Lab3/best_model.joblib, Lab3/scaler_params.json Lab4/
python Lab4/demo.py --batch
```

Для ручного ввода показаний запусти demo.py без --batch.
