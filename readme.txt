Порядок запуска:

1. all_pre_process.py
Основной препроцессинг данных.
Читает короткие JSON-файлы из all_data_cut_14800, переводит int16-коды АЦП в Вольты, применяет полосовой фильтр вокруг 2.95 МГц, выполняет IQ-демодуляцию, НЧ-фильтрацию и децимацию.
На выходе сохраняет .pt-тензоры формы (2, 3, 148) в pre_processed_data/train и pre_processed_data/test.

2. graph_hist.py
Строит гистограммы СКО комплексной огибающей сигнала соленоида для train и test.
Нужен для проверки масштаба данных и распределения уровней помехи.

3. train.py
Обучает модель (архитектура в shield_free_model.py) на данных из pre_processed_data/train.
Данные нормируются на 5 мВ. Модель учится предсказывать помеху в соленоиде по сигналу внешнего датчика.
Лучшая модель и история loss сохраняются в models_save/ver_1.

4. loss_curve.py
Строит график train loss и validation loss по файлу models_save/ver_1/loss_history.npz.

5. graph.py
Строит временные сигналы и спектры.
При MODEL_ON = False показывает TARGET FID и TARGET Spectrum.
При MODEL_ON = True показывает TARGET, PRED и CLEAN = TARGET - PRED.


Другие файлы:

SpecDecode.py
Читает исходный JSON: достаёт base64, переводит его в int16, затем масштабирует в Вольты по диапазону PicoScope.

new_dataset.py
Загружает .pt-файлы и возвращает:
features — внешний датчик;
target — соленоид.
После DataLoader форма данных: [B, 2, 1, 148].

shield_free_model.py
Архитектура CNN-модели.
Вход: [B, 2, 1, 148].
Выход: [B, 2, 1, 148].