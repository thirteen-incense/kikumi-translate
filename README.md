# Kikumi Translate Worker

音频转写工作节点，配合 Kikumi 翻译服务端使用。负责拉取音频转写任务，**并行执行中日文语音识别**，并用 **LLM 精调**中文字幕，最终提交 LRC 格式字幕结果。

## 工作原理

```
Kikumi 服务端  ←→  Kikumi Translate Worker（本程序）
                        │
                        ├─ 1. 注册工作节点并登录（CSRF 认证）
                        ├─ 2. 轮询获取转写任务
                        ├─ 3. 下载音频文件
                        ├─ 4. 语音活动检测 (VAD)
                        ├─ 5. 并行转录
                        │       ├─ 日中直译模型 → 中文直译 LRC
                        │       └─ 日文 ASR 模型 → 日文 LRC（两遍精调）
                        ├─ 6. LLM 精调中文字幕
                        │      （以日文时间轴为基准，参考中文直译润色）
                        └─ 7. 提交 LRC 字幕 → 回服务端
```

1. **注册与登录** — 向 Kikumi 服务端注册工作节点名称，并从登录页面获取 CSRF token，提交表单以获取 session。
2. **轮询任务** — 每隔 `POLL_INTERVAL` 秒调用 API 获取待处理的转写任务。
3. **下载音频** — 将服务端的音频文件下载到本地缓存目录。若 session 失效会自动重新登录。
4. **语音活动检测 (VAD)** — 使用 Silero VAD 模型检测音频中的人声片段，过滤静音部分。中日文可设置独立的 VAD 策略。
5. **并行语音识别** — 使用 `multiprocessing.ProcessPoolExecutor`（`spawn` 模式）同时启动两个 mlx-whisper 实例：
   - **日中直译模型** — 中文字幕直出
   - **日文 ASR 模型** — 日文转录，对密集语段执行**第二遍精调**以获得更精确的时间戳
6. **LLM 精调中文字幕** — 以日文 LRC 的时间轴为基准，参考中文直译内容，通过 LLM（DeepSeek 远程 API 或本地模型）对中文字幕进行润色优化。
7. **提交结果** — 将精调后的 LRC 内容通过 API 提交回 Kikumi 服务端，清理内存缓存和本地临时文件。

> **中断恢复** — 任务处理期间会缓存任务信息到 `task_cache.json`。如果进程意外退出，重启后会自动检测并继续未完成的任务。

> **调试缓存** — 设置 `SAVE_INTERMEDIATE_LRC=true` 后，中间结果会保存到 `cache/debug/` 目录；若该目录下已存在中日字幕文件，下次处理相同音频时**跳过转录，直接进行 LLM 精调**。

## 环境要求

- **macOS**（依赖 Apple Silicon 优化的 mlx-whisper）
- **Python 3.10+**
- 已运行的 Kikumi 服务端实例
- MLX 格式的 Whisper 模型文件（见下文）

## 安装

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt
```

### 依赖列表

| 包 | 用途 |
|---|---|
| `requests` | HTTP 客户端，与 Kikumi API 通信 |
| `mlx-whisper` | Apple Silicon 上运行的 Whisper 模型 |
| `numpy` | 音频数据处理 |
| `soundfile` | 写入临时 WAV 文件供 mlx-whisper 读取 |
| `onnxruntime` | Silero VAD 模型推理 |
| `av` (PyAV) | 音频解码（支持多种格式） |

额外工具依赖（`convert2mlx.py`）：

| 包 | 用途 | 安装方式 |
|---|---|---|
| `torch` | 加载原始 PyTorch Whisper 权重 | `pip install torch` |
| `mlx` | MLX 框架核心库 | `pip install mlx` |
| `tqdm` | 下载进度条 | `pip install tqdm` |
| `huggingface_hub` | 从 Hugging Face 下载 / 上传模型（可选） | `pip install huggingface_hub` |

### 模型文件

**VAD 模型**已内置在 `assets/silero_vad.onnx`。

**Whisper 模型**需要额外下载（mlx-whisper 格式），放置在 `cache/mlx_models/` 目录下：

```bash
# 日中直译模型（路径参考，可通过 MLX_WHISPER_MODEL_PATH 自定义）
cache/mlx_models/whisper-large-v2-translate-zh-v0.2-st-mlx/

# 日文 ASR 模型（路径参考，可通过 MLX_WHISPER_MODEL_PATH_JA 自定义）
cache/mlx_models/whisper-ja-1.5B-mlx/
```

模型可通过 `convert2mlx.py` 工具从 Hugging Face 转换得到（见下文工具章节）。

## 配置

所有配置通过环境变量进行：

### 基础配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `KIKUMI_URL` | `http://192.168.10.5:4000` | Kikumi 服务端地址 |
| `KIKUMI_USERNAME` | `admin` | 登录用户名 |
| `KIKUMI_PASSWORD` | `admin` | 登录密码 |
| `WORKER_NAME` | `mac_01` | 工作节点名称（用于注册和标识） |
| `POLL_INTERVAL` | `5` | 轮询间隔（秒） |

### 模型 & 转录配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MLX_WHISPER_MODEL_PATH` | `cache/mlx_models/whisper-large-v2-translate-zh-v0.2-st-mlx` | 日中直译模型路径 |
| `MLX_WHISPER_MODEL_PATH_JA` | `cache/mlx_models/whisper-ja-1.5B-mlx` | 日文 ASR 模型路径 |
| `TRANSCRIBE_LANGUAGE` | `ja` | 转录语言（如 `ja`、`zh`、`en`） |
| `LOCAL_AUDIO_DIR` | `./cache/input` | 音频下载缓存目录 |
| `MAX_WORKERS` | `2` | 并行转录进程数（中日文各一个） |

### VAD 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `VAD_MODE` | `low1` | 中文直译模型的 VAD 策略 |
| `VAD_MODE_JA` | 同 `VAD_MODE` | 日文 ASR 模型的独立 VAD 策略 |

**VAD 模式说明：**

| 模式 | 说明 |
|---|---|
| `high` | 阈值高，只检测较明显的语音，适合安静环境 |
| `low` | 平衡漏检和误检 |
| `low1` | **（默认）** 阈值更低，静音段容忍更长，适合语速慢、停顿多的内容 |
| `low_zh` | 中文优化模式 |
| `low_ja` | 日语优化模式 |
| 不设置或设为其他值 | 跳过 VAD，整个音频作为一个片段处理 |

### 流水线控制

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `FULL_PIPELINE` | `true` | `true`=执行完整流水线（日中直译→日文ASR→LLM）；`false`=仅日中直出，跳过日文转录及 LLM 精调 |
| `SAVE_INTERMEDIATE_LRC` | `false` | 是否保留中间字幕文件（中文/日文/精调后中间结果），调试用。启用后还会从 `cache/debug/` 加载已有缓存跳过转录 |
| `HEARTBEAT_INTERVAL` | `3` | 心跳间隔（秒），长任务期间保持 session 有效 |

### LLM 精调配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_MODE` | `auto` | `local`=本地模型 / `remote`=远端 API；`auto` 则向后兼容：有 `LLM_API_KEY` 走 remote，否则走 local |
| `LLM_CONCURRENT` | `true` | 多批 LLM 调用是否并行执行（`ThreadPoolExecutor`）；设为 `false` 则串行减少 API 并发压力 |

**远程 API 模式（`LLM_MODE=remote` 或 `auto` + 有 API_KEY）：**

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_API_KEY` | — | API 密钥，填入后启用远程 API |
| `LLM_API_URL` | `https://api.deepseek.com/v1/chat/completions` | API 端点（兼容 OpenAI 格式） |
| `LLM_MODEL_NAME` | `deepseek-v4-flash` | 模型名称 |

**本地模型模式（`LLM_MODE=local` 或 `auto` + 无 API_KEY）：**

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_LOCAL_API_URL` | `http://localhost:11434/v1/chat/completions` | 本地 API 端点 |
| `LLM_LOCAL_MODEL_NAME` | `qwen2.5:7b` | 本地模型名称 |
| `LLM_LOCAL_API_KEY` | — | 本地模型若需要 API Key（如使用 lms 等代理时） |

## 使用

```bash
# 方式一：通过 run.sh 启动（自带环境变量配置）
# 编辑 run.sh 修改配置后运行
bash run.sh

# 方式二：直接运行，通过环境变量传参
source .venv/bin/activate
export KIKUMI_URL="http://192.168.8.5:4000"
export KIKUMI_USERNAME="admin"
export KIKUMI_PASSWORD="kikumi"
export WORKER_NAME="mac_01"
export MLX_WHISPER_MODEL_PATH="./cache/mlx_models/whisper-large-v2-translate-zh-v0.2-st-mlx"
export MLX_WHISPER_MODEL_PATH_JA="./cache/mlx_models/whisper-ja-1.5B-mlx"
export FULL_PIPELINE="true"
export LLM_MODE="remote"
export LLM_API_KEY="sk-xxx"
export LLM_API_URL="https://api.deepseek.com/v1/chat/completions"
export LLM_MODEL_NAME="deepseek-v4-flash"
python worker.py
```

启动后程序会：
1. 连接到 Kikumi 服务端并注册
2. 登录获取 session（自动重试直至成功）
3. 设置 MLX 内存上限（16 GB）并打印 LLM 配置状态
4. 检查是否有遗留的未完成任务（中断恢复）
5. 进入轮询循环，等待新任务
6. 每获取到一个任务就执行：下载 → VAD → 并行转录 → LLM 精调 → 提交 → 清理

## 流水线详解

### 并行转录

使用 `multiprocessing.ProcessPoolExecutor`（`spawn` 模式）同时启动两个 mlx-whisper 实例：

- **Worker 1** — 加载日中直译模型，输出中文 LRC
- **Worker 2** — 加载日文 ASR 模型，输出日文 LRC

两个转录独立进行，互不等待。进程数通过 `MAX_WORKERS` 控制。

### 日文两遍转录

日文 ASR 采用两遍转录策略以提升密集语段的准确性：

1. **第一遍** — 粗粒度 VAD 切分 → mlx-whisper 转录 → 句子切分
2. **第二遍（精调）** — 对包含多句的密集语段，用细粒度 VAD 重新切分（`threshold=0.2`），配合 `word_timestamps=True` 重新转录，获得精确的逐句时间戳

### LLM 精调

精调模块以日文 LRC 的时间轴为基准，参考中文直译 LRC 的内容，通过 LLM 对中文字幕进行语义润色：

1. 将 LRC 时间戳替换为序号以减少 token 消耗
2. 携带 system prompt（`transcribe/refine-prompt.md`）调用 LLM API
3. 解析 LLM 返回的 JSON，校验行数一致性
4. 当日文字幕行数超过 `BATCH_SIZE`（30 行）时，自动分批；`LLM_CONCURRENT=true` 时多批并行处理
5. 失败自动重试（最多 5 次）

系统 prompt 可在 `transcribe/refine-prompt.md` 中编辑，修改后即时生效，无需重启。

### 音频下载容错

下载音频时如果服务器返回 302 重定向（session 失效），会自动重新登录再重试下载。

### 提交重试

提交字幕结果到服务端时若遇到连接异常，会每 5 秒自动重试，直至成功。

### 心跳机制

长任务期间使用心跳线程（每 `HEARTBEAT_INTERVAL` 秒）保持 session 有效，并记录心跳失败/恢复状态。

### 内存管理

每个任务完成后自动执行：

- `mx.set_memory_limit(16 GB)` — 在启动时设置 MLX 内存上限
- `mx.set_cache_limit(0)` + `mx.clear_cache()` — 清理 MLX Metal 缓存
- 创建临时张量触发显式回收，再恢复缓存上限至 512 MB
- `gc.collect()` — 强制 Python GC
- 删除本地临时文件（音频 + 字幕缓存 + 任务缓存）

## 项目结构

```
kikumi-translate/
├── worker.py                     # 主入口：轮询 + 任务编排 + 中断恢复
├── server/
│   ├── __init__.py
│   └── kikumi_api.py             # Kikumi API 客户端（CSRF 登录、注册、取任务、下载、提交、心跳、状态更新）
├── transcribe/
│   ├── __init__.py
│   ├── audio.py                  # 音频解码（PyAV → float32 16kHz mono）
│   ├── mlx_whisper.py            # 语音识别（VAD + mlx-whisper + LRC 生成 + 两遍精调）
│   ├── vad.py                    # Silero VAD 实现（ONNX Runtime）
│   ├── refine.py                 # LLM 精调模块（远程 API / 本地模型，并发分批）
│   └── refine-prompt.md          # LLM 精调用 system prompt（可热编辑）
├── convert2mlx.py                # 独立工具：转换 PyTorch Whisper → MLX 格式
├── assets/
│   └── silero_vad.onnx           # 预训练的 Silero VAD 模型
├── cache/
│   ├── input/                    # 音频文件下载缓存
│   ├── mlx_models/               # mlx-whisper 模型存放目录
│   ├── debug/                    # 中间字幕缓存（SAVE_INTERMEDIATE_LRC 时生成）
│   └── db/                       # token / session 持久化
├── requirements.txt              # Python 依赖
├── run.sh                        # 启动脚本（含环境变量配置示例）
└── README.md                     # 本文件
```

## 工具脚本

### convert2mlx.py

将 Hugging Face 上的 PyTorch Whisper 模型转换为 MLX 格式，以便 mlx-whisper 加载。

```bash
# 日中直译模型：chickenrice0721/whisper-large-v2-translate-zh-v0.2-st
python convert2mlx.py \
    --torch-name-or-path chickenrice0721/whisper-large-v2-translate-zh-v0.2-st \
    --mlx-path mlx_models/whisper-large-v2-translate-zh-v0.2-st-mlx

# 日文 ASR 模型：efwkjn/whisper-ja-1.5B
python convert2mlx.py \
    --torch-name-or-path efwkjn/whisper-ja-1.5B \
    --mlx-path mlx_models/whisper-ja-1.5B-mlx
```

