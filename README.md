# YanTian Global Medium-Range Weather Forecasting Inference Model

## 1. Project Overview

**YanTian** is a large-scale AI model designed for **global medium-range weather forecasting**. The model is built upon the proposed **Searth Transformer (Shifted Earth Transformer)** architecture and incorporates the novel **Relay Autoregressive (RAR)** fine-tuning strategy, enabling high-accuracy medium-range forecasting under relatively limited computational resources.

This repository releases the **inference model (ONNX format) and real-time inference scripts**, which support rolling global 6-hour forecasts driven by real-time GFS data.

------

## 2. Paper Information

### Title

**Searth Transformer: A Transformer Architecture Incorporating Earth's Geospheric Physical Priors for Global Mid-Range Weather Forecasting**

### Paper Link

https://doi.org/10.48550/arXiv.2601.09467

------

### Core Contributions

#### 1️⃣ Searth Transformer (Shifted Earth Transformer)

- Explicitly incorporates physical priors of the Earth system into window-based self-attention:
  - Meridional non-periodic boundaries (North–South boundaries)
  - Zonal periodic continuity (East–West periodicity)
- Introduces an asymmetric shift-and-mask mechanism:
  - Removes zonal boundary masks to enable global longitudinal information exchange
  - Preserves polar masks to prevent physically unrealistic cross-pole mixing
- Significantly improves large-scale atmospheric circulation modeling capability

------

#### 2️⃣ Relay Autoregressive (RAR) Fine-Tuning Strategy

- Decomposes long-horizon autoregressive forecasting into multiple sub-stages
- Performs independent backpropagation within each stage
- Applies gradient detachment between stages
- Substantially reduces GPU memory consumption
- Enables learning of continuous atmospheric evolution up to 15 days

------

#### 3️⃣ YanTian Model Performance

- Resolution: 1°
- Number of variables: 69 atmospheric variables
- Temporal resolution: 6 hours
- Z500 skillful forecast lead time reaches **10.3 days**
- At 1° resolution:
  - Outperforms ECMWF HRES
  - Achieves performance comparable to state-of-the-art AI models
- Peak training GPU memory usage < 25 GB

------

## 3. Model Specifications

| Item                  | Description              |
| --------------------- | ------------------------ |
| Model Name            | YanTian                  |
| Architecture          | Encoder–Core–Decoder     |
| Transformer           | Searth Transformer       |
| Parameters            | ~600M                    |
| Input Time Steps      | 2 historical states      |
| Output Time Step      | 1 future 6-hour forecast |
| Horizontal Resolution | 1° (180 × 360)           |
| Time Interval         | 6 hours                  |
| Number of Variables   | 69                       |

------

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

The normalization and 4×4 window-averaging downscaling procedures follow the implementation in the  prepare.py 、download.py .

Standardization formula:
$$
X_{norm} = \frac{X - \mu}{\sigma}
$$
where:

- $X$ = raw meteorological field
- $\mu$ = mean from `statistics.json`
- $\sigma$ = standard deviation from `statistics.json`

------

### Model Input

$$
x \in \mathbb{R}^{B \times 2 \times 69 \times 180 \times 360}
$$

Dimension definitions:

| Dimension | Meaning                             |
| --------- | ----------------------------------- |
| B         | Batch size                          |
| 2         | Two historical time steps (t-6h, t) |
| 69        | Number of meteorological variables  |
| 180       | Latitude (H)                        |
| 360       | Longitude (W)                       |

Tensor layout:

```
(Batch, Time, Channel, Lat, Lon)
```

------

### Model Output

$$
\hat{y} \in \mathbb{R}^{B \times 69 \times 180 \times 360}
$$

The output represents:

- The next 6-hour forecast
- 69 variables
- 1° global grid

Tensor layout:

```
(Batch, Channel, Lat, Lon)
```

------

### 69 Atmospheric Variables

#### Pressure Level Order

```
[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] hPa
```

------

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

------

#### Surface Variables (4 channels)

1. **U10** – 10 m zonal wind
   Index: 65
2. **V10** – 10 m meridional wind
   Index: 66
3. **T2M** – 2 m temperature
   Index: 67
4. **MSL** – Mean sea level pressure
   Index: 68

------

## 5. Project Structure

Both Google drive (https://drive.google.com/drive/folders/1DhDZR79buQYBOBTY2ini_Q4_bVY2DJ5C?usp=sharing) and Baidu disk (https://pan.baidu.com/s/1NpDJLqNMjlNcK8ic-ZPbzA?pwd=7dad) contain the YanTian model. Please download the model from the cloud drive and place it in the folder.

```
YanTian/
│── run_cpu.py      # CPU-based inference using real-time GFS data
│── prepare_data.py      # Normalization and 4×4 window-averaging downscaling
│── download.py     # GFS data download script
│── YanTian.onnx    # ONNX inference model
│── YanTian.data    # ONNX weight data file
│── statistics.json    # Stores the mean and variance of meteorological elements
│── environment.txt    
└── README.md
```

## 6. Project Startup and Execution

### Pre-Launch Configuration

#### 1. Modify `start_time_str`

In the `main()` function of `run_cpu.py`, update the parameter:

```
start_time_str
```

Set it to one of the four synoptic cycles (00, 06, 12, 18 UTC) within the past three days.

⚠️ Make sure the selected time is available according to the official GFS data release schedule.

------

#### 2. Modify `time_length`

In the `main()` function of `run_cpu.py`, update the parameter:

```
time_length
```

This parameter controls the number of autoregressive inference steps performed by the model (i.e., how many future 6-hour forecasts will be generated).

------

#### 3. Configure the environment according to the `environment.txt` file.


### Launch Command

Run the following command:

```
python run_cpu.py
```

------

### Output Description

#### 1. Downloaded and Processed Input Data

After execution:

- Downloaded GFS raw data
- Preprocessed (normalized and downscaled) input data

will be stored in:

```
down_load/
```

------

#### 2. Forecast Results

The prediction outputs will be saved in:

```
predict/
```

The result is stored as a single data tensor with shape:
$$
(time\_length + 1,\ 69,\ 180,\ 360)
$$

- Index `0` → Atmospheric state at the initial time `start_time_str`
- Index `1` → Forecast at +6h
- Index `2` → Forecast at +12h
- ...
- Index `time_length` → Final forecast time

Thus, the tensor contains:

- 1 initial state
- `time_length` future forecast states
- Each state includes 69 variables on a 1° global grid (180 × 360)

# -------------------中文版-------------------

# YanTian 全球中期天气预报推理模型

## 一、项目简介

**YanTian（演天）** 是一个面向全球中期天气预报（Medium-range Weather Forecasting）的 AI 大模型。该模型基于提出的 **Searth Transformer（Shifted Earth Transformer）** 架构，并结合创新的 **Relay Autoregressive (RAR)** 微调策略，实现了在较低计算资源条件下的高精度中期天气预报能力。

本项目开源内容为 **推理模型（ONNX 格式）及实时推理脚本**，可用于基于实时 GFS 数据进行全球 6 小时间隔滚动预报。

------

## 二、论文信息

### 论文标题

**Searth Transformer: A Transformer Architecture Incorporating Earth's Geospheric Physical Priors for Global Mid-Range Weather Forecasting**

### 论文链接

https://doi.org/10.48550/arXiv.2601.09467

### 论文核心贡献

本文提出：

### 1️⃣ Searth Transformer（Shifted Earth Transformer）

- 在 Transformer 的 window-based self-attention 中显式引入：
  - 经向非周期边界（南北边界）
  - 纬向周期连续性（东西方向周期）
- 通过非对称 shift-and-mask 机制：
  - 取消经向周期 mask，实现全球经向信息连续传播
  - 保留极区 mask，避免物理不合理的跨极信息混合
- 提升了大尺度环流建模能力

### 2️⃣ Relay Autoregressive (RAR) 微调策略

- 将长时间滚动预测分解为多个子阶段
- 每阶段独立反向传播
- 阶段间进行梯度 detach
- 显著降低 GPU 显存占用
- 支持学习 15 天连续演变

### 3️⃣ YanTian 模型性能

- 分辨率：1°
- 变量数：69 个气象变量
- 时间分辨率：6 小时
- Z500 技能时效达到 **10.3 天**
- 在 1° 分辨率下：
  - 超越 ECMWF HRES
  - 达到当前主流 AI 模型水平
- 训练峰值显存 < 25GB

## 三、模型基本信息

| 项目        | 内容                 |
| ----------- | -------------------- |
| 模型名称    | YanTian              |
| 架构        | Encoder-Core-Decoder |
| Transformer | Searth Transformer   |
| 参数量      | ~600M                |
| 输入时间步  | 2 个历史时刻         |
| 输出时间步  | 1 个未来 6h 预报     |
| 水平分辨率  | 1° (180 × 360)       |
| 时间间隔    | 6 小时               |
| 变量数      | 69                   |

## 四、输入数据说明

### 输入数据预处理

YanTian 推理模型对输入数据的物理变量顺序、空间排布方式以及归一化方式有严格要求。如不符合下述规范，模型预测结果将出现严重偏差。

模型输入前必须使用 `statistics.json` 文件中提供的 **69 个变量对应的均值（mean）和标准差（std）进行标准化处理**，同时需要对输入变量水平空间分辨率降尺度从0.25度降低到1度，维度从（721，1440）变为（180，360），（由于采用窗口平均风发721无法被4整除，因此舍弃最后的南极点，从721变为180）。归一化和降尺度窗口平均操作和参考 prepare.py 、download.py 

标准化方式：
$$
X_{norm} = \frac{X - \mu}{\sigma}
$$
其中：

- $X$ 为原始气象场
- $\mu$ 为 `statistics.json` 中对应变量的均值
- $\sigma$ 为 `statistics.json` 中对应变量的标准差

### 输入

$$
x \in \mathbb{R}^{B \times 2 \times 69 \times 180 \times 360}
$$

维度含义：

| 维度 | 含义                      |
| ---- | ------------------------- |
| B    | Batch size                |
| 2    | 两个历史时间步（t-6h, t） |
| 69   | 气象变量通道数            |
| 180  | 纬度（H）                 |
| 360  | 经度（W）                 |

即：

```
(Batch, Time, Channel, Lat, Lon)
```

------

### 输出

$$
\hat{y} \in \mathbb{R}^{B \times 69 \times 180 \times 360}
$$

输出为：

- 下一个 6 小时的预报场
- 69 个变量
- 1° 全球网格

即：

```
(Batch, Channel, Lat, Lon)
```

### 69个变量

高度层顺序为[50,100,150,200,250,300,400,500,600,700,850,925,1000]

1. Z（位势而非位势高度，单位m2 s-2），索引：z50-z1000: 0-12
2. R（相对湿度，单位%），索引：r50-r1000: 13-25
3. T（开尔文温度，单位K），索引：t50-t1000: 26-38
4. U（东西风），索引：u50-u1000: 39-51
5. V（南北风），索引：v50-v1000: 52-64

1. U10：（10米高度东西风），索引：65
2. V10：（10米高度南北风），索引：66
3. T2M：（两米温度），索引：67
4. MSL：（海平面气压），索引：68

## 五、项目文件结构说明

谷歌云盘 (https://drive.google.com/drive/folders/1DhDZR79buQYBOBTY2ini_Q4_bVY2DJ5C?usp=sharing) 和百度网盘 (https://pan.baidu.com/s/1NpDJLqNMjlNcK8ic-ZPbzA?pwd=7dad) 都包含 YanTian 模型。请从云盘下载模型并将其放置在文件夹中。

```
YanTian/
│── run_cpu.py      # 基于CPU的推理脚本（使用实时GFS数据）
│── prepare.py      # 数据归一化处理
│── download.py     # GFS数据下载+4×4窗口平均降尺度脚本
│── YanTian.onnx    # ONNX格式的推理模型文件
│── YanTian.data    # ONNX模型的权重数据文件
│── statistics.json    # 存储气象要素均值与方差
│── environment.txt    
└── README.md          # 项目说明文档
```

## 六、项目启动与运行

### 启动前参数配置：

1、修改run_cpu.py文件中main函数中的start_time_str参数，改为近3天的00、06、12、18四个时刻的时间点，注意根据GFS官网数据实时更新为准。

2、修改run_cpu.py文件中main函数中time_length参数，这个参数控制模型自回归推理多少步

3、根据environment.txt  文件配置环境



### 启动命令行：

```python
python run_cpu.py
```



### 结果保存：

1、运行推理文件后，下载并与处理好的输入数据会存储在down_load/文件夹内

2、预报结果存储在predict/文件夹中，存储为一个数据矩阵，矩阵中包括1个起报时刻start_time_str和未来time_length个预报时刻的气象数据，结果矩阵维度为（time_length+1，69，180，360），索引0为起报时刻start_time_str的气象要素。

模型训练:**李田野**（中国科学院大气物理研究所）等，RUMLA。
<img width="344" height="344" alt="8d8101fcf37782ab5120fdb51ce797ad" src="https://github.com/user-attachments/assets/96a85599-af7f-43c1-89dd-f22c00d94f63" />

如有问题或需要预训练权重，欢迎通过 GitHub Issues 联系。

