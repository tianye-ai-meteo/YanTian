"""
YanTian System — Global Medium-Range Weather Forecasting Inference.

Pipeline:
  GFS 0.25° GRIB → 4×4 window average → 1°
  → normalize → YanTian Forecast ONNX → 1° forecast (normalized)
  → YanTian Downscale ONNX → 0.25° forecast (normalized)
  → denormalize → save .npy (0.25°)

Components:
  - YanTian Forecast Model (预报模型):  Searth Transformer + RAR, 1° (ONNX)
  - YanTian Downscale Model (降尺度模型): MetDownscaler, 0.25° super-resolution (ONNX)

Usage:
  python inference.py
"""

import os
import json
import numpy as np
import onnxruntime as ort
from datetime import datetime, timedelta

from download import download_files, process_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
FORECAST_ONNX = os.path.join(ROOT, "YanTian_forecast.onnx")
DOWNSCALE_ONNX = os.path.join(ROOT, "YanTian_downscaler.onnx")
STATS_PATH = os.path.join(ROOT, "statistics.json")
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "predict")

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def load_normalization_stats():
    """Load 69-channel normalization statistics from statistics.json.

    statistics.json contains 85 entries. Extract 69 channels matching
    the YanTian variable order: z(13)->r(13)->t(13)->u(13)->v(13)->u10->v10->t2m->msl.

    Returns:
        avg_69, std_69: (69,) float32 arrays
    """
    with open(STATS_PATH) as f:
        stats = json.load(f)

    avg = np.array(stats["avg"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    # Pressure: indices 7:-13 (z, r, t, u, v x 13 levels = 65)
    pressure_avg = avg[7:-13]
    pressure_std = std[7:-13]

    # Surface: u10, v10, t2m, msl (skip d2m at index 2)
    surface_avg = np.concatenate([avg[0:2], avg[3:5]])
    surface_std = np.concatenate([std[0:2], std[3:5]])

    return np.concatenate([pressure_avg, surface_avg]), \
           np.concatenate([pressure_std, surface_std])


def load_and_normalize_input(date_str):
    """Load 1-degree GFS data for two consecutive timesteps and normalize.

    Args:
        date_str: str, format YYYYMMDDHH

    Returns:
        np.ndarray, (1, 2, 69, 180, 360) float32 normalized
    """
    avg_69, std_69 = load_normalization_stats()
    avg_69 = avg_69[:, None, None]
    std_69 = std_69[:, None, None]

    current_dt = datetime.strptime(date_str, "%Y%m%d%H")
    past_dt = current_dt - timedelta(hours=6)

    tensors = []
    for dt in [past_dt, current_dt]:
        dt_str = dt.strftime("%Y%m%d%H")
        npy_path = os.path.join(DATA_DIR, dt_str, f"gfs_{dt_str}.npy")
        data = np.load(npy_path).astype(np.float32)  # (69, 180, 360)
        data = (data - avg_69) / std_69
        tensors.append(data)

    input_arr = np.stack(tensors, axis=0)     # (2, 69, 180, 360)
    input_arr = input_arr[np.newaxis, ...]    # (1, 2, 69, 180, 360)
    return input_arr


def denormalize(data, avg_69, std_69):
    """Denormalize 69-channel data back to physical units.

    Args:
        data: (..., 69, H, W) normalized array
        avg_69, std_69: (69,) arrays

    Returns:
        Same shape as data, physical values
    """
    reshape = [1] * data.ndim
    reshape[-3] = 69
    return data * std_69.reshape(reshape) + avg_69.reshape(reshape)


# ---------------------------------------------------------------------------
# ONNX Loader
# ---------------------------------------------------------------------------

def load_onnx(path):
    """Load an ONNX model with optimal execution provider."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4

    preferred = ["CUDAExecutionProvider", "ROCMExecutionProvider"]
    available = ort.get_available_providers()
    providers = [p for p in preferred if p in available] + ["CPUExecutionProvider"]

    session = ort.InferenceSession(path, sess_options=opts, providers=providers)
    print(f"  Loaded: {os.path.basename(path)} ({session.get_providers()[0]})")
    return session


# ---------------------------------------------------------------------------
# Inference Pipeline
# ---------------------------------------------------------------------------

def run_inference(start_time, time_length):
    """Run the full YanTian system inference pipeline.

    Pipeline:
      GFS 0.25 degree -> 4x4 window average -> 1 degree -> normalize
      -> YanTian Forecast ONNX -> 1 degree normalized forecast
      -> YanTian Downscale ONNX -> 0.25 degree normalized forecast
      -> denormalize -> save (.npy, 0.25 degree)

    Args:
        start_time: str, YYYYMMDDHH (e.g. "2026030306")
        time_length: int, number of 6-hour forecast steps (e.g. 40 = 10 days)

    Returns:
        str, path to the saved 0.25-degree forecast .npy file
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load models
    print("[YanTian] Loading models...")
    forecast_session = load_onnx(FORECAST_ONNX)
    downscale_session = load_onnx(DOWNSCALE_ONNX)

    forecast_input_name = forecast_session.get_inputs()[0].name
    downscale_input_name = downscale_session.get_inputs()[0].name

    # 2. Load normalization stats
    avg_69, std_69 = load_normalization_stats()

    # 3. Download GFS data (start time + previous 6h for autoregressive input)
    print(f"[YanTian] Downloading GFS data for {start_time}...")
    for dt in [start_time,
               (datetime.strptime(start_time, "%Y%m%d%H") - timedelta(hours=6)).strftime("%Y%m%d%H")]:
        download_files(folder=os.path.join(DATA_DIR, ""), date_time=dt)
        process_data(folder=os.path.join(DATA_DIR, ""), date_time=dt)

    # 4. Load and normalize input
    print("[YanTian] Preparing input data...")
    input_arr = load_and_normalize_input(start_time)  # (1, 2, 69, 180, 360)

    # 5. Autoregressive forecast + downscale loop
    forecasts_025 = []

    for step in range(time_length):
        # --- Step A: YanTian Forecast Model (1-degree ONNX) ---
        # Input:  (1, 2, 69, 180, 360)
        # Output: (1, 69, 180, 360) normalized
        output_1deg = forecast_session.run(
            None, {forecast_input_name: input_arr}
        )[0]

        # --- Step B: YanTian Downscale Model (0.25-degree ONNX) ---
        # Input:  (1, 69, 180, 360)
        # Output: (1, 69, 721, 1440) normalized
        output_025deg = downscale_session.run(
            None, {downscale_input_name: output_1deg}
        )[0]

        # --- Step C: Denormalize and store 0.25-degree result ---
        forecast_phys = denormalize(output_025deg[0], avg_69, std_69)  # (69, 721, 1440)
        forecasts_025.append(forecast_phys)

        # --- Step D: Update autoregressive input ---
        input_arr[:, 0, :, :, :] = input_arr[:, 1, :, :, :]  # past = previous present
        input_arr[:, 1, :, :, :] = output_1deg                # present = new forecast

        print(f"  Step {step + 1}/{time_length} completed.")

    # 6. Save 0.25-degree forecast
    out = np.stack(forecasts_025, axis=0)  # (time_length, 69, 721, 1440)

    start_dt = datetime.strptime(start_time, "%Y%m%d%H")
    end_dt = start_dt + timedelta(hours=6 * time_length)
    out_name = f"{start_time}_to_{end_dt.strftime('%Y%m%d%H')}_025deg.npy"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    np.save(out_path, out)

    print(f"\n[YanTian] Forecast saved: {out_path}")
    print(f"[YanTian] Shape: {out.shape}")
    print("[YanTian] Done!")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # === Configuration ===
    START_TIME = "2026030306"   # YYYYMMDDHH
    TIME_LENGTH = 2             # Number of 6-hour steps (40 = 10 days)

    run_inference(START_TIME, TIME_LENGTH)
