# YanTian System — Global Medium-Range Weather Forecasting

## 1. Project Overview

**YanTian System** is a complete global medium-range weather forecasting system consisting of two components:

- **YanTian Forecast Model**: Built upon the **Searth Transformer (Shifted Earth Transformer)** architecture with **Relay Autoregressive (RAR)** fine-tuning, producing 1° global forecasts (~600M parameters).
- **YanTian Downscale Model**: A lightweight **MetDownscaler** that super-resolves the 1° forecast to 0.25° resolution (~29M parameters).

The system achieves **10.3-day skillful Z500 forecast lead time** and outputs **0.25° high-resolution forecasts** suitable for detailed weather analysis.

This repository releases the **inference pipeline (ONNX format)** for both models, supporting rolling global 6-hour forecasts driven by real-time GFS data.

---

## 2. Paper Information

### Title

**Searth Transformer: A Transformer Architecture Incorporating Earth's Geospheric Physical Priors for Global Mid-Range Weather Forecasting**

### Paper Link

https://doi.org/10.48550/arXiv.2601.09467

---

### Core Contributions

#### 1️⃣ Searth Transformer (Shifted Earth Transformer)

- Explicitly incorporates physical priors of the Earth system into window-based self-attention:
  - Meridional non-periodic boundaries (North–South boundaries)
  - Zonal periodic continuity (East–West periodicity)
- Introduces an asymmetric shift-and-mask mechanism:
  - Removes zonal boundary masks to enable global longitudinal information exchange
  - Preserves polar masks to prevent physically unrealistic cross-pole mixing
- Significantly improves large-scale atmospheric circulation modeling capability

---

#### 2️⃣ Relay Autoregressive (RAR) Fine-Tuning Strategy

- Decomposes long-horizon autoregressive forecasting into multiple sub-stages
- Performs independent backpropagation within each stage
- Applies gradient detachment between stages
- Substantially reduces GPU memory consumption
- Enables learning of continuous atmospheric evolution up to 15 days

---

#### 3️⃣ YanTian System Performance

- Forecast resolution: 1° (180 × 360) → downscaled to 0.25° (721 × 1440)
- Number of variables: 69 atmospheric variables
- Temporal resolution: 6 hours
- Z500 skillful forecast lead time reaches **10.3 days**
- At 1° resolution: outperforms ECMWF HRES, achieves performance comparable to state-of-the-art AI models
- Peak training GPU memory usage < 25 GB

---

## 3. Model Specifications

### YanTian Forecast Model (1°)

| Item | Description |
|------|-------------|
| Architecture | Encoder–Core–Decoder |
| Transformer | Searth Transformer |
| Parameters | ~600M |
| Input | 2 historical states (B, 2, 69, 180, 360) |
| Output | 1 future 6h forecast (B, 69, 180, 360) |
| Resolution | 1° (180 × 360) |

### YanTian Downscale Model (0.25°)

| Item | Description |
|------|-------------|
| Architecture | Dual-path residual (bilinear baseline + learned residual) |
| Blocks | 12 × Hybrid (RCA + spatial attention) |
| Parameters | ~29M |
| Upsampling | Progressive 2-stage PixelShuffle (×2 → ×2) |
| Input | (B, 69, 180, 360) |
| Output | (B, 69, 721, 1440) |

---

## 4. Input Data Description

### Input Data Preprocessing

The YanTian inference model requires strict adherence to:

- Variable ordering
- Spatial arrangement
- Normalization procedure

Failure to follow the specifications below will result in severe forecast degradation.

Before inference, input data must be standardized using the **mean and standard deviation of the 69 variables provided in `statistics.json`**.

Additionally:

- The input spatial resolution must be downscaled from **0.25° to 1°**
- The grid dimension must change from **(721, 1440) → (180, 360)**

Since 721 cannot be evenly divided by 4 when applying 4×4 window averaging, the last Antarctic latitude row is discarded, resulting in 180 latitudes.

The normalization and 4×4 window-averaging downscaling procedures follow the implementation in `download.py`.

Standardization formula:
$$
X_{norm} = \frac{X - \mu}{\sigma}
$$
where:

- $X$ = raw meteorological field
- $\mu$ = mean from `statistics.json`
- $\sigma$ = standard deviation from `statistics.json`

---

### Model Input

$$
x \in \mathbb{R}^{B \times 2 \times 69 \times 180 \times 360}
$$

Dimension definitions:

| Dimension | Meaning |
|-----------|---------|
| B | Batch size |
| 2 | Two historical time steps (t-6h, t) |
| 69 | Number of meteorological variables |
| 180 | Latitude (H) |
| 360 | Longitude (W) |

Tensor layout:

```
(Batch, Time, Channel, Lat, Lon)
```

---

### Forecast Model Output (1°)

$$
\hat{y}_{1°} \in \mathbb{R}^{B \times 69 \times 180 \times 360}
$$

### Downscale Model Output (0.25°)

$$
\hat{y}_{0.25°} \in \mathbb{R}^{B \times 69 \times 721 \times 1440}
$$

The system ultimately produces a **0.25° high-resolution forecast**.

---

### 69 Atmospheric Variables

#### Pressure Level Order

```
[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] hPa
```

#### Upper-Air Variables (5 × 13 = 65 channels)

1. **Z** – Geopotential (NOT geopotential height), unit: m² s⁻²
   Index: z50–z1000 → 0–12
2. **R** – Relative Humidity, unit: %
   Index: r50–r1000 → 13–25
3. **T** – Temperature, unit: Kelvin (K)
   Index: t50–t1000 → 26–38
4. **U** – Zonal wind (east–west component)
   Index: u50–u1000 → 39–51
5. **V** – Meridional wind (north–south component)
   Index: v50–v1000 → 52–64

#### Surface Variables (4 channels)

1. **U10** – 10 m zonal wind
   Index: 65
2. **V10** – 10 m meridional wind
   Index: 66
3. **T2M** – 2 m temperature
   Index: 67
4. **MSL** – Mean sea level pressure
   Index: 68

---

## 5. Project Structure

```
YanTian/
│── inference.py                     # Main entry: forecast + downscale pipeline
│── YanTian_forecast.onnx            # YanTian Forecast Model (1°, ONNX)
│── YanTian_forecast.data            # Forecast model weight data
│── YanTian_downscaler.onnx          # YanTian Downscale Model (0.25°, ONNX)
│── YanTian_downscaler.onnx.data     # Downscale model weight data
│── download.py                      # GFS 0.25° download + 4×4 window averaging
│── prepare_data.py                  # Data normalization utilities
│── statistics.json                  # 69-channel normalization mean/std
│── environment.txt                  # Conda environment specification
│── data/                            # Downloaded & processed GFS data (auto-created)
│── predict/                         # Forecast output (auto-created)
└── README.md
```

---

## 6. Inference Pipeline

```
GFS 0.25° GRIB  ──[4×4 window avg]──▶  1° raw
    │
    ▼  normalize (statistics.json)
(1, 2, 69, 180, 360)
    │
    ▼  YanTian Forecast Model (ONNX)
(1, 69, 180, 360)  ──  1° normalized forecast
    │
    ▼  YanTian Downscale Model (ONNX)
(1, 69, 721, 1440)  ──  0.25° normalized forecast
    │
    ▼  denormalize
(69, 721, 1440)  ──  physical units
    │
    ▼  save
predict/{start}_to_{end}_025deg.npy
```

---

## 7. Setup

### Environment

Both models share the same ONNX Runtime environment; no deep learning framework (PyTorch/TensorFlow) is required for inference.

**Option A — Using conda (recommended):**

```bash
conda create --name yantian python=3.10
conda activate yantian
pip install numpy onnxruntime pygrib xarray netcdf4
```

**Option B — Using the provided environment.txt:**

```bash
conda env create -f environment.txt
conda activate weather-forecast
```

### Model Files

Download ONNX model files from the cloud drive and place them in the project root:

| File | Source |
|------|--------|
| `YanTian_forecast.onnx` | [Google Drive](https://drive.google.com/drive/folders/1DhDZR79buQYBOBTY2ini_Q4_bVY2DJ5C?usp=sharing) / [Baidu Pan](https://pan.baidu.com/s/1NpDJLqNMjlNcK8ic-ZPbzA?pwd=7dad) |
| `YanTian_forecast.data` | Same cloud drive |
| `YanTian_downscaler.onnx` | Same cloud drive |
| `YanTian_downscaler.onnx.data` | Same cloud drive |

---

## 8. Usage

### Configuration

Edit `inference.py` to set the start time and forecast length:

```python
START_TIME = "2026030306"   # YYYYMMDDHH (choose a recent time)
TIME_LENGTH = 40            # 40 steps × 6h = 10-day forecast
```

### Run

```bash
python inference.py
```

### Output

The 0.25° high-resolution forecast is saved as a `.npy` file in the `predict/` directory:

```
predict/{start_time}_to_{end_time}_025deg.npy
```

Shape: `(time_length, 69, 721, 1440)`

- Index `0` → Forecast at +6h
- Index `1` → Forecast at +12h
- ...
- Index `time_length - 1` → Final forecast time

Each time step contains 69 atmospheric variables on a 0.25° global grid (721 × 1440).

---

## Citation

```bibtex
@article{li2025searth,
  title={Searth Transformer: A Transformer Architecture Incorporating Earth's Geospheric Physical Priors for Global Mid-Range Weather Forecasting},
  author={Li, Tianye and others},
  journal={arXiv preprint arXiv:2601.09467},
  year={2025}
}
```

---

# -------------------中文版-------------------

# YanTian 系统 — 全球中期天气预报

## 一、项目简介

**YanTian 系统（演天系统）** 是一个完整的全球中期天气预报系统，由两个组件构成：

- **YanTian 预报模型**：基于 **Searth Transformer（Shifted Earth Transformer）** 架构和 **Relay Autoregressive (RAR)** 微调策略，输出 1° 分辨率预报（~600M 参数）。
- **YanTian 降尺度模型**：轻量级 **MetDownscaler**，将 1° 预报超分辨率提升至 0.25°（~29M 参数）。

系统在 1° 分辨率下 Z500 技能时效达到 **10.3 天**，最终输出 **0.25° 高分辨率预报场**。

本仓库开源 **推理管线（ONNX 格式）**，支持基于实时 GFS 数据进行全球 6 小时间隔滚动预报。

| 术语 | English | 中文 |
|------|---------|------|
| YanTian System | YanTian System | YanTian 系统 |
| YanTian Forecast Model | YanTian Forecast Model | YanTian 预报模型 |
| YanTian Downscale Model | YanTian Downscale Model | YanTian 降尺度模型 |

---

## 二、论文信息

### 论文标题

**Searth Transformer: A Transformer Architecture Incorporating Earth's Geospheric Physical Priors for Global Mid-Range Weather Forecasting**

### 论文链接

https://doi.org/10.48550/arXiv.2601.09467

### 论文核心贡献

#### 1️⃣ Searth Transformer（Shifted Earth Transformer）

- 在 Transformer 的 window-based self-attention 中显式引入：
  - 经向非周期边界（南北边界）
  - 纬向周期连续性（东西方向周期）
- 通过非对称 shift-and-mask 机制：
  - 取消经向周期 mask，实现全球经向信息连续传播
  - 保留极区 mask，避免物理不合理的跨极信息混合
- 提升了大尺度环流建模能力

#### 2️⃣ Relay Autoregressive (RAR) 微调策略

- 将长时间滚动预测分解为多个子阶段
- 每阶段独立反向传播
- 阶段间进行梯度 detach
- 显著降低 GPU 显存占用
- 支持学习 15 天连续演变

#### 3️⃣ YanTian 系统性能

- 预报分辨率：1°（180 × 360）→ 降尺度至 0.25°（721 × 1440）
- 变量数：69 个气象变量
- 时间分辨率：6 小时
- Z500 技能时效达到 **10.3 天**
- 在 1° 分辨率下超越 ECMWF HRES，达到当前主流 AI 模型水平
- 训练峰值显存 < 25 GB

---

## 三、模型规格

### YanTian 预报模型（1°）

| 项目 | 说明 |
|------|------|
| 架构 | Encoder-Core-Decoder |
| Transformer | Searth Transformer |
| 参数量 | ~600M |
| 输入 | 2 个历史时刻 (B, 2, 69, 180, 360) |
| 输出 | 1 个未来 6h 预报 (B, 69, 180, 360) |
| 分辨率 | 1° (180 × 360) |

### YanTian 降尺度模型（0.25°）

| 项目 | 说明 |
|------|------|
| 架构 | 双路径残差（双线性基线 + 学习残差） |
| 模块 | 12 × Hybrid（RCA + 空间注意力） |
| 参数量 | ~29M |
| 上采样 | 渐进式 2 阶段 PixelShuffle（×2 → ×2） |
| 输入 | (B, 69, 180, 360) |
| 输出 | (B, 69, 721, 1440) |

---

## 四、输入数据说明

### 输入数据预处理

YanTian 推理模型对输入数据的物理变量顺序、空间排布方式以及归一化方式有严格要求。如不符合下述规范，模型预测结果将出现严重偏差。

模型输入前必须使用 `statistics.json` 文件中提供的 **69 个变量对应的均值（mean）和标准差（std）进行标准化处理**，同时需要对输入变量水平空间分辨率从 0.25° 降尺度到 1°，维度从（721，1440）变为（180，360）（由于采用窗口平均法，721 无法被 4 整除，因此舍弃最后的南极行，从 721 变为 180）。归一化和窗口平均操作参考 `download.py`。

标准化方式：
$$
X_{norm} = \frac{X - \mu}{\sigma}
$$
其中：

- $X$ 为原始气象场
- $\mu$ 为 `statistics.json` 中对应变量的均值
- $\sigma$ 为 `statistics.json` 中对应变量的标准差

### 模型输入

$$
x \in \mathbb{R}^{B \times 2 \times 69 \times 180 \times 360}
$$

维度含义：

| 维度 | 含义 |
|------|------|
| B | Batch size |
| 2 | 两个历史时间步（t-6h, t） |
| 69 | 气象变量通道数 |
| 180 | 纬度（H） |
| 360 | 经度（W） |

即：

```
(Batch, Time, Channel, Lat, Lon)
```

### 预报模型输出（1°）

$$
\hat{y}_{1°} \in \mathbb{R}^{B \times 69 \times 180 \times 360}
$$

### 降尺度模型输出（0.25°）

$$
\hat{y}_{0.25°} \in \mathbb{R}^{B \times 69 \times 721 \times 1440}
$$

系统最终输出 **0.25° 高分辨率预报场**。

### 69 个变量

高度层顺序为 [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

1. Z（位势而非位势高度，单位 m² s⁻²），索引：z50–z1000: 0–12
2. R（相对湿度，单位 %），索引：r50–r1000: 13–25
3. T（开尔文温度，单位 K），索引：t50–t1000: 26–38
4. U（东西风），索引：u50–u1000: 39–51
5. V（南北风），索引：v50–v1000: 52–64
6. U10（10 米高度东西风），索引：65
7. V10（10 米高度南北风），索引：66
8. T2M（两米温度），索引：67
9. MSL（海平面气压），索引：68

---

## 五、项目文件结构

```
YanTian/
│── inference.py                     # 主入口：预报 + 降尺度管线
│── YanTian_forecast.onnx            # YanTian 预报模型（1°, ONNX）
│── YanTian_forecast.data            # 预报模型权重数据
│── YanTian_downscaler.onnx          # YanTian 降尺度模型（0.25°, ONNX）
│── YanTian_downscaler.onnx.data     # 降尺度模型权重数据
│── download.py                      # GFS 0.25° 下载 + 4×4 窗口平均
│── prepare_data.py                  # 数据归一化工具
│── statistics.json                  # 69 通道归一化均值/标准差
│── environment.txt                  # Conda 环境配置
│── data/                            # 下载并处理后的 GFS 数据（自动创建）
│── predict/                         # 预报输出（自动创建）
└── README.md
```

---

## 六、推理管线

```
GFS 0.25° GRIB  ──[4×4 窗口平均]──▶  1° 原始场
    │
    ▼  归一化 (statistics.json)
(1, 2, 69, 180, 360)
    │
    ▼  YanTian 预报模型 (ONNX)
(1, 69, 180, 360)  ──  1° 归一化预报
    │
    ▼  YanTian 降尺度模型 (ONNX)
(1, 69, 721, 1440)  ──  0.25° 归一化预报
    │
    ▼  反归一化
(69, 721, 1440)  ──  物理单位
    │
    ▼  保存
predict/{start}_to_{end}_025deg.npy
```

---

## 七、环境配置

### 方法一 — 使用 conda（推荐）

```bash
conda create --name yantian python=3.10
conda activate yantian
pip install numpy onnxruntime pygrib xarray netcdf4
```

### 方法二 — 使用提供的 environment.txt

```bash
conda env create -f environment.txt
conda activate weather-forecast
```

### 模型文件

从云盘下载 ONNX 模型文件并放置在项目根目录：

| 文件 | 来源 |
|------|------|
| `YanTian_forecast.onnx` | [Google Drive](https://drive.google.com/drive/folders/1DhDZR79buQYBOBTY2ini_Q4_bVY2DJ5C?usp=sharing) / [百度网盘](https://pan.baidu.com/s/1NpDJLqNMjlNcK8ic-ZPbzA?pwd=7dad) |
| `YanTian_forecast.data` | 同上 |
| `YanTian_downscaler.onnx` | 同上 |
| `YanTian_downscaler.onnx.data` | 同上 |

---

## 八、使用说明

### 配置参数

编辑 `inference.py` 中的参数：

```python
START_TIME = "2026030306"   # YYYYMMDDHH（选择近期的起报时刻）
TIME_LENGTH = 40            # 40 步 × 6h = 10 天预报
```

### 启动命令

```bash
python inference.py
```

### 结果保存

0.25° 高分辨率预报结果保存在 `predict/` 目录中：

```
predict/{start_time}_to_{end_time}_025deg.npy
```

维度：`(time_length, 69, 721, 1440)`

- 索引 `0` → 起报后 +6h 预报
- 索引 `1` → 起报后 +12h 预报
- ...
- 索引 `time_length - 1` → 最终预报时刻

每个时刻包含 69 个变量在 0.25° 全球网格（721 × 1440）上的预报场。

---

模型训练：**李田野**（中国科学院大气物理研究所）等，RUMLA。
<img width="344" height="344" alt="8d8101fcf37782ab5120fdb51ce797ad" src="https://github.com/user-attachments/assets/96a85599-af7f-43c1-89dd-f22c00d94f63" />

如有问题或需要预训练权重，欢迎通过 GitHub Issues 联系。
