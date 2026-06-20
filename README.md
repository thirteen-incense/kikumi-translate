# Kikumi Translate Worker

一个音频转写工作节点，配合 Kikumi 翻译服务端使用。负责拉取音频转写任务，**并行执行中日文语音识别**，并用 **LLM 精调** 中文字幕，最终提交 LRC 格式字幕结果。

## 工作原理

```
Kikumi 服务端  ←→  Kikumi Translate Worker（本程序）
                        │
                        ├─ 1. 轮询获取转写任务
                        ├─ 2. 下载音频文件
                        ├─ 3. 语音活动检测 (VAD)
                        ├─ 4. 并行转写
                        │       ├─ ja_zh 模型 → 中文直译 LRC
                        │       └─ large-v3 模型 → 日文 LRC（两遍精调）
                        ├─ 5. LLM 精调中文字幕
                        │      （以日文时间轴为基准，参考中文直译润色）
                        └─ 6. 提交 LRC 字幕 → 回服务端
```

1. **注册与登录** — 向 Kikumi 服务端注册工作节点名称，并通过 Web 表单登录获取 session。
2. **轮询任务** — 每隔 `POLL_INTERVAL` 秒调用 API 获取待处理的转写任务。
3. **下载音频** — 将服务端的音频文件下载到本地缓存目录。
4. **语音活动检测 (VAD)** — 使用 Silero VAD 模型检测音频中的人声片段，过滤静音部分。支持多种 VAD 策略（常规、ASMR、中日文独立策略）。
5. **并行语音识别** — 同时启动两个 mlx-whisper 实例：
   - **ja_zh 模型** — 日中直译（输出中文 LRC）
   - **large-v3 模型** — 日文 ASR（输出日文 LRC），对密集语段执行**第二遍精调**以获得更精确的时间戳
6. **LLM 精调中文字幕** — 以日文 LRC 的时间轴为基准，参考中文直译内容，通过 LLM（DeepSeek 远程 API 或本地模型）对中文字幕进行润色优化。
7. **提交结果** — 将精调后的 LRC 内容通过 API 提交回 Kikumi 服务端，清理内存缓存和本地临时文件。

> **中断恢复** — 任务处理期间会缓存任务信息到 `task_cache.json`。如果进程意外退出，重启后会自动检测并继续未完成的任务。

## 环境要求

- **macOS**（依赖 Apple Silicon 优化的 mlx-whisper）
- **Python 3.10+**
- 已运行的 Kikumi 服务端实例
- （可选）DeepSeek / OpenAI 兼容 API 密钥，或本地推理服务

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
# 中文直译模型（默认路径，可通过 MLX_WHISPER_MODEL_PATH 自定义）
cache/mlx_models/whisper-large-v2-translate-zh-v0.2-st-mlx/

# 日文 ASR 模型（默认路径，可通过 MLX_WHISPER_MODEL_PATH_JA 自定义）
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

### VAD 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `VAD_MODE` | `low` | 中文直译模型的 VAD 策略 |
| `VAD_MODE_JA` | 同 `VAD_MODE` | 日文 ASR 模型的独立 VAD 策略 |

**VAD 模式说明：**

| 模式 | 说明 |
|---|---|
| `high` | 阈值高，只检测较明显的语音，适合安静环境 |
| `low` | 默认模式，平衡漏检和误检 |
| `low1` | 阈值更低，静音段容忍更长，适合语速慢、停顿多的内容 |
| `low_zh` | 中文优化模式 |
| `low_ja` | 日语优化模式 |
| `asmr` | ASMR 内容优化模式 |
| 不设置或设为其他值 | 跳过 VAD，整个音频作为一个片段处理 |

### 流水线控制

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `FULL_PIPELINE` | `true` | `true`=执行完整流水线（ja_zh→large-v3→LLM）；`false`=仅 ja_zh 日中直出，跳过日文转录及 LLM 精调 |
| `SAVE_INTERMEDIATE_LRC` | `false` | 是否保留中间字幕文件（中文/日文中间结果），调试用 |
| `HEARTBEAT_INTERVAL` | `3` | 心跳间隔（秒），长任务期间保持 session 有效 |

### LLM 精调配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_MODE` | （自动） | `local`=本地模型 / `remote`=远端 API；不设置则向后兼容：有 `LLM_API_KEY` 走 remote，否则走 local |

**远程 API 模式（`LLM_MODE=remote`）：**

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_API_KEY` | — | API 密钥，填入后启用远程 API |
| `LLM_API_URL` | `https://api.deepseek.com/v1/chat/completions` | API 端点 |
| `LLM_MODEL_NAME` | `deepseek-chat` | 模型名称 |

**本地模型模式（`LLM_MODE=local`）：**

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_LOCAL_API_URL` | `http://localhost:11434/v1/chat/completions` | 本地 API 端点 |
| `LLM_LOCAL_MODEL_NAME` | `qwen2.5:7b` | 本地模型名称 |

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
export LLM_API_URL="https://api.deepseek.com/chat/completions"
export LLM_MODEL_NAME="deepseek-v4-flash"
python worker.py
```

启动后程序会：
1. 连接到 Kikumi 服务端并注册
2. 设置 MLX 内存限制（16 GB）
3. 检查是否有遗留的未完成任务（中断恢复）
4. 进入轮询循环，等待新任务
5. 每获取到一个任务就执行：下载 → VAD → 并行转录 → LLM 精调 → 提交 → 清理

## 流水线详解

### 并行转录

使用 `multiprocessing.ProcessPoolExecutor`（`spawn` 模式）同时启动两个 mlx-whisper 实例：

- **Worker 1** — 加载日中直译模型（如 `whisper-large-v2-translate-zh-v0.2-st-mlx`），输出中文 LRC
- **Worker 2** — 加载日文 ASR 模型（如 `whisper-ja-1.5B-mlx`），输出日文 LRC

两个转录独立进行，互不等待。

### 日文两遍转录

日文 ASR 采用两遍转录策略以提升密集语段的准确性：

1. **第一遍** — 粗粒度 VAD 切分 → mlx-whisper 转录 → 句子切分
2. **第二遍（精调）** — 对包含多句的密集语段，用细粒度 VAD 重新切分，配合 `word_timestamps=True` 重新转录，获得精确的逐句时间戳

### LLM 精调

精调模块以日文 LRC 的时间轴为基准，参考中文直译 LRC 的内容，通过 LLM 对中文字幕进行语义润色：

1. 将 LRC 时间戳替换为序号以减少 token 消耗
2. 携带系统 prompt（`transcribe/refine-prompt.md`）调用 LLM API
3. 解析 LLM 返回的 JSON，校验行数一致性
4. 失败自动重试（最多 5 次）

系统 prompt 可在 `transcribe/refine-prompt.md` 中编辑，修改后即时生效，无需重启。

### 内存管理

每个任务完成后自动执行：
- `mx.clear_cache()` + `mx.set_cache_limit(0)` 清理 MLX Metal 缓存
- `gc.collect()` 强制 Python GC
- 删除本地临时文件

## 项目结构

```
kikumi-translate/
├── worker.py                     # 主入口：轮询 + 任务编排
├── server/
│   ├── __init__.py
│   └── kikumi_api.py             # Kikumi API 客户端（登录、注册、取任务、下载、提交、心跳）
├── transcribe/
│   ├── __init__.py
│   ├── audio.py                  # 音频解码（PyAV → float32 16kHz mono）
│   ├── mlx_whisper.py            # 语音识别（VAD + mlx-whisper + LRC 生成 + 两遍精调）
│   ├── vad.py                    # Silero VAD 实现（ONNX Runtime）
│   ├── refine.py                 # LLM 精调模块（DeepSeek / 本地模型）
│   └── refine-prompt.md          # LLM 精调用的 system prompt
├── convert2mlx.py                # 独立工具：转换 PyTorch Whisper → MLX 格式
├── assets/
│   └── silero_vad.onnx           # 预训练的 Silero VAD 模型
├── cache/
│   ├── input/                    # 音频文件下载缓存
│   └── mlx_models/               # mlx-whisper 模型存放目录
├── requirements.txt              # Python 依赖
├── run.sh                        # 启动脚本（含环境变量配置）
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

