import os
import torch
import numpy as np
from scipy.signal import decimate, butter, sosfilt
from tqdm import tqdm
from scr.preprocess.SpecDecode import DataDecoder
import re

RANGE_CODE_TO_V = {
    0: 10e-3,
    1: 20e-3,
    2: 50e-3,
    3: 100e-3,
    4: 200e-3,
    5: 500e-3,
}

def parse_range_codes(file_path):
    # ищет _xx в названиях для правильно перевода в вольты
    name = os.path.splitext(os.path.basename(file_path))[0]
    m = re.search(r'_(\d)(\d)(?:_s)?$', name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def apply_decimation(array):
    # децимация 
    # 14800 точек → 1480 точек → 148 точек, т.е. снижаем частоту до 80 кГц
    result = decimate(array, 10, axis=1)
    result = decimate(result, 10, axis=1)
    result = decimate(result, 1, axis=1)
    return result

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    path = "./all_data_cut_14800/" # путь к папке с файлами сигналов
    save_path = "./pre_processed_data/" # папка для сохранения тензоров
    print("CWD:", os.getcwd())
    print("save_path:", os.path.abspath(save_path))

    # рандомно делит данные на train/test в отношении 90/10
    train_ratio = 0.90
    rng = np.random.default_rng(0)

    ensure_dir(save_path)
    ensure_dir(os.path.join(save_path, "train"))
    ensure_dir(os.path.join(save_path, "test"))

    # основные параметры
    center_freq = 2.95e6 # центральная частота 2.95 МГц
    low_freq = 40e3 # полоса после демодуляции 40 кГц
    window_points = 14800 # длина до децимации

    json_files = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".json"):
                json_files.append(os.path.join(root, f))

    print("Found json files:", len(json_files))

    train_idx = 0
    test_idx = 0

    # обработка каждого файла
    for file_path in tqdm(json_files):
        print("Processing:", file_path)

        # достаем диапазон из названия
        codes = parse_range_codes(file_path)
        if codes is None:
            print("SKIP: cannot parse range codes from filename:", file_path)
            continue

        v0 = RANGE_CODE_TO_V.get(codes[0])
        v1 = RANGE_CODE_TO_V.get(codes[1])
        if v0 is None or v1 is None:
            print("SKIP: bad range codes:", codes, file_path)
            continue

        rel_dir = os.path.relpath(os.path.dirname(file_path), path)
        train_out_dir = os.path.join(save_path, "train", rel_dir)
        test_out_dir = os.path.join(save_path, "test", rel_dir)
        ensure_dir(train_out_dir)
        ensure_dir(test_out_dir)

        dec = DataDecoder(file_path)

        averaging_num = 0
        data_num = 0

        points_raw = dec.getDataPoints(averaging_num, data_num)
        points_declared = int(points_raw[0]) if isinstance(points_raw, (list, tuple)) else int(points_raw)

        rate = dec.getDataRate(averaging_num, data_num)

        points_num = points_declared # т.к. уже порезанные на маленькие кусочки


        # создаем коэф. для полосового фильтра 2.85–3.05 МГц
        dt = 1.0 / rate
        fs = rate

        lower_freq = center_freq - 0.1e6
        higher_freq = center_freq + 0.1e6
        filter = butter(4, [lower_freq, higher_freq], 'bp', fs=fs, output='sos')

        # создаем опорные сигналы для IQ-демодуляции
        time = np.arange(0, window_points * dt, dt)
        real_part = np.sin(2 * np.pi * center_freq * time)
        imaginary_part = np.cos(2 * np.pi * center_freq * time)

        # создаем коэф. для НЧ-фильтра
        low_pass_filter = butter(4, low_freq, 'lowpass', fs=fs, output='sos')

        # base64 → int16 → Вольты и сохраняем данные соленоида и внешнего датчика
        target_1d = dec.getDataScaled(averaging_num, data_num, 0, points_num, v_range=v0)
        ch1_1d = dec.getDataScaled(averaging_num, data_num, 1, points_num, v_range=v1)

        if len(ch1_1d) != window_points or len(target_1d) != window_points:
            print("SKIP: bad signal length:", len(ch1_1d), len(target_1d), file_path)
            continue

        ch1 = ch1_1d.reshape(1, window_points)
        target = target_1d.reshape(1, window_points)

        # полосовой фильтр
        ch1 = sosfilt(filter, ch1, axis=1)
        target = sosfilt(filter, target, axis=1)

        # IQ-демодуляция
        # сигнал переносится из области около 2.95 МГц в низкочастотную область
        ch1_real = ch1 * real_part
        ch1_imag = ch1 * imaginary_part
        target_real = target * real_part
        target_imag = target * imaginary_part

        # НЧ-фильтр
        # оставляется низкочастотная часть до 40 кГц
        ch1_real = sosfilt(low_pass_filter, ch1_real, axis=1)
        ch1_imag = sosfilt(low_pass_filter, ch1_imag, axis=1)
        target_real = sosfilt(low_pass_filter, target_real, axis=1)
        target_imag = sosfilt(low_pass_filter, target_imag, axis=1)

        # децимация
        ch1_real = apply_decimation(ch1_real)
        ch1_imag = apply_decimation(ch1_imag)
        target_real = apply_decimation(target_real)
        target_imag = apply_decimation(target_imag)

        if ch1_real.shape[1] != 148:
            print("SKIP: bad decimation length:", ch1_real.shape)
            continue

        # сохраняем тензоры
        for i in range(ch1_real.shape[0]):
            data = torch.zeros((2, 3, 148))
            data[0, 0, :] = torch.from_numpy(ch1_real[i].copy())
            data[1, 0, :] = torch.from_numpy(ch1_imag[i].copy())
            data[0, 2, :] = torch.from_numpy(target_real[i].copy())
            data[1, 2, :] = torch.from_numpy(target_imag[i].copy())

            if rng.random() < train_ratio:
                out_path = os.path.join(train_out_dir, f"{train_idx}.pt")
                train_idx += 1
            else:
                out_path = os.path.join(test_out_dir, f"{test_idx}.pt")
                test_idx += 1

            torch.save(data, out_path)

    print("DONE")
    print("train samples:", train_idx)
    print("test samples :", test_idx)

if __name__ == '__main__':
    main()

