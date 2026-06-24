import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

from code.model.shield_free_model import Net1


# CONFIG

ROOT_TEST = "./pre_processed_data/test"
MODEL_PATH = "./models_save/ver_1/best.pth"

N_EXAMPLES = 100
SLICE_WINDOW = 10
SEED = 1

# После препроцесса: 8 МГц / 100 = 80 кГц
ORIGINAL_RATE_HZ = 8_000_000.0
DECIMATION_COEF = 100
DT_NEW = (1.0 / ORIGINAL_RATE_HZ) * DECIMATION_COEF

# Фильтрация примеров по СКО сигнала
STD_RANGE_MV = (0.0, 10000.0)
STD_CHANNEL_IDX = 2  # соленоид

# Что строить
PLOT_ALL_ON_ONE = True
MODEL_ON = True

# Спектр
SPECTRUM_TO_MV = True       # True: амплитуда спектра в мВ
SPECTRUM_CENTER = False      # убрать DC-компоненту перед FFT
CROP_START = 0              # если нужно отрезать начало перед FFT


def list_pt_files(root):
    files = []

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".pt"):
                files.append(os.path.join(dirpath, fn))

    files.sort()
    return files


def load_complex_channel_from_pt(pt_path, channel_idx=0, slice_window=0, device="cpu"):
    """
    Загружает один канал из .pt и возвращает комплексный сигнал I + jQ.

    .pt имеет форму:
        data.shape = (2, 3, 148)

    Ось 0:
        0 -> I / real
        1 -> Q / imag

    Ось 1:
        0 -> внешний датчик / input
        2 -> соленоид / target

    Ось 2:
        время
    """
    data = torch.load(pt_path, map_location=device)

    if data.shape[0] != 2 or data.shape[1] < 3:
        raise ValueError(f"Bad tensor shape {data.shape} in file {pt_path}")

    I = data[0, channel_idx, :].detach().cpu().numpy().astype(np.float64)
    Q = data[1, channel_idx, :].detach().cpu().numpy().astype(np.float64)

    if slice_window > 0:
        I = I[slice_window:-slice_window]
        Q = Q[slice_window:-slice_window]

    return I + 1j * Q


def center_complex(x):
    return x - np.mean(x)


def complex_rms_mv(x):
    """
    СКО комплексной огибающей в мВ:
        sqrt(mean(I^2 + Q^2)) * 1e3
    """
    rms_v = np.sqrt(np.mean(np.real(x) ** 2 + np.imag(x) ** 2))
    return float(rms_v * 1e3)


def spectrum_complex(x_complex_1d, dt_new, center=True, to_mv=True):
    """
    Строит спектр комплексного IQ-сигнала.

    X:
        частота после IQ-демодуляции, кГц;
        это отстройка от центральной частоты 2.95 МГц, а не абсолютная частота.

    Y:
        если to_mv=True: нормированная амплитуда спектра, мВ;
        если to_mv=False: нормированная амплитуда спектра, В.
    """
    x = np.asarray(x_complex_1d)

    if CROP_START > 0 and CROP_START < len(x):
        x = x[CROP_START:]

    n = len(x)
    if n < 2:
        return None, None

    if center:
        x = center_complex(x)

    freq_khz = fftshift(np.fft.fftfreq(n, d=dt_new)) / 1000.0

    sp = np.abs(fftshift(fft(x))) / n

    if to_mv:
        sp = sp * 1e3

    return freq_khz, sp


def complex_to_2ch(x_complex_1d):
    """
    np.complex[L] -> torch.Tensor[1, 2, L]
    """
    x2 = np.stack([np.real(x_complex_1d), np.imag(x_complex_1d)], axis=0)
    return torch.from_numpy(x2).unsqueeze(0).float()


def to_model_input_4d(x_1_2_L):
    """
    [B, 2, L] -> [B, 2, 1, L]
    """
    return x_1_2_L.unsqueeze(2)


def slice_time_torch(x, slice_window):
    """
    Режет последнюю ось, то есть время.
    Работает для [B, 2, L] и [B, 2, 1, L].
    """
    if slice_window <= 0:
        return x
    return x[..., slice_window:x.shape[-1] - slice_window]


def center_torch_time(x):
    """
    Центрирует по времени отдельно каждую компоненту I/Q.
    """
    return x - torch.mean(x, dim=-1, keepdim=True)


def torch_2ch_to_complex(x):
    """
    Принимает [1, 2, L] или [1, 2, 1, L].
    Возвращает np.complex[L].
    """
    if x.ndim == 4:
        x = x.squeeze(2)

    x_np = x.detach().cpu().numpy()[0]
    return x_np[0] + 1j * x_np[1]


def plot_one_model_example(pt_path, model, device):
    """
    Строит TARGET / PRED / CLEAN для одного примера:
        PRED  = предсказанная моделью помеха в соленоиде
        CLEAN = TARGET - PRED
    """
    x_in = load_complex_channel_from_pt(pt_path, channel_idx=0, slice_window=0)
    x_tg = load_complex_channel_from_pt(pt_path, channel_idx=2, slice_window=0)

    scale_constant = 5e-3  # как при обучении

    features = complex_to_2ch(x_in).to(device)
    features = to_model_input_4d(features)

    with torch.no_grad():
        pred = model(features / scale_constant)

    pred = pred * scale_constant

    pred = slice_time_torch(pred, SLICE_WINDOW)
    pred_c = torch_2ch_to_complex(pred)

    x_in_cut_raw = x_in[SLICE_WINDOW:-SLICE_WINDOW]
    x_tg_cut_raw = x_tg[SLICE_WINDOW:-SLICE_WINDOW]

    x_clean_raw = x_tg_cut_raw - pred_c

    x_tg_cut_plot = center_complex(x_tg_cut_raw)
    pred_c_plot = center_complex(pred_c)
    x_clean_plot = center_complex(x_clean_raw)

    # FID
    plt.plot(np.real(x_tg_cut_plot) * 1e3, alpha=0.8, label="TARGET real")
    plt.plot(np.real(pred_c_plot) * 1e3, alpha=0.8, label="PRED real")
    plt.plot(np.real(x_clean_plot) * 1e3, alpha=0.8, label="CLEAN real")
    plt.xlabel("Отсчёт")
    plt.ylabel("Амплитуда, мВ")
    plt.title("Один пример: временной сигнал")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # SPECTRUM
    f_in, sp_in = spectrum_complex(x_in_cut_raw, DT_NEW, center=False, to_mv=SPECTRUM_TO_MV)
    f_tg, sp_tg = spectrum_complex(x_tg_cut_raw, DT_NEW, center=False, to_mv=SPECTRUM_TO_MV)
    f_pr, sp_pr = spectrum_complex(pred_c, DT_NEW, center=False, to_mv=SPECTRUM_TO_MV)
    f_cl, sp_cl = spectrum_complex(x_clean_raw, DT_NEW, center=False, to_mv=SPECTRUM_TO_MV)

    plt.figure(figsize=(12, 5))
    # plt.plot(f_in, sp_in, alpha=0.8, label="INPUT")
    plt.plot(f_tg, sp_tg, alpha=0.8, label="TARGET")
    plt.plot(f_pr, sp_pr, alpha=0.8, label="PRED")
    plt.plot(f_cl, sp_cl, alpha=0.8, label="CLEAN")
    plt.xlabel("Отстройка от 2.95 МГц, кГц")
    plt.ylabel("Амплитуда, мВ" if SPECTRUM_TO_MV else "Амплитуда, В")
    plt.title("Один пример: спектр")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = None
    if MODEL_ON:
        model = Net1().to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()

    pt_files = list_pt_files(ROOT_TEST)

    if not pt_files:
        raise RuntimeError(f"No .pt files found under: {ROOT_TEST}")

    # фильтрация по СКО target 
    lo_mv, hi_mv = STD_RANGE_MV

    filtered = []
    stds_mv = []

    for p in pt_files:
        x_tg = load_complex_channel_from_pt(
            p,
            channel_idx=STD_CHANNEL_IDX,
            slice_window=SLICE_WINDOW,
        )

        s_mv = complex_rms_mv(x_tg)

        if lo_mv <= s_mv <= hi_mv:
            filtered.append(p)
            stds_mv.append(s_mv)

    if not filtered:
        raise RuntimeError(
            f"No .pt files match STD_RANGE_MV={STD_RANGE_MV} mV under: {ROOT_TEST}"
        )

    chosen = random.sample(filtered, k=min(N_EXAMPLES, len(filtered)))

    print("STD_RANGE_MV:", STD_RANGE_MV)
    print("Total files:", len(pt_files))
    print("Filtered files:", len(filtered))
    print("Chosen examples:", len(chosen))

    print("Filtered STD, mV:")
    print("  min   :", np.min(stds_mv))
    print("  median:", np.median(stds_mv))
    print("  mean  :", np.mean(stds_mv))
    print("  max   :", np.max(stds_mv))

    if MODEL_ON:
        plot_one_model_example(chosen[0], model, device)

    all_fid_tg = []
    all_sp_tg = []
    freq_ref = None

    for pt_path in chosen:
        x_tg = load_complex_channel_from_pt(
            pt_path,
            channel_idx=2,
            slice_window=SLICE_WINDOW,
        )

        x_tg_for_plot = center_complex(x_tg)

        freq_tg, sp_tg = spectrum_complex(
            x_tg,
            DT_NEW,
            center=False,
            to_mv=SPECTRUM_TO_MV,
        )

        if PLOT_ALL_ON_ONE:
            all_fid_tg.append(x_tg_for_plot)
            all_sp_tg.append(sp_tg)

            if freq_ref is None:
                freq_ref = freq_tg

    if PLOT_ALL_ON_ONE:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        for x in all_fid_tg:
            plt.plot(np.real(x) * 1e3, alpha=0.35)

        plt.title("TARGET FID")
        plt.xlabel("Отсчёт")
        plt.ylabel("Амплитуда, мВ")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        for sp in all_sp_tg:
            plt.plot(freq_ref, sp, alpha=0.35)

        plt.title("TARGET Spectrum")
        plt.xlabel("Отстройка от 2.95 МГц, кГц")
        plt.ylabel("Амплитуда, мВ" if SPECTRUM_TO_MV else "Амплитуда, В")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()