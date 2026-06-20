#!/bin/bash
source .venv/bin/activate

export KIKUMI_URL="http://192.168.10.6:4000"
export KIKUMI_USERNAME="admin"
export KIKUMI_PASSWORD="kikumi"
export WORKER_NAME="mac_01"
export MLX_WHISPER_MODEL_PATH="./cache/mlx_models/whisper-large-v2-translate-zh-v0.2-st-mlx"
export MLX_WHISPER_MODEL_PATH_JA="./cache/mlx_models/whisper-ja-1.5B-mlx"
export LOCAL_AUDIO_DIR="./cache/input"
export TRANSCRIBE_LANGUAGE="ja"
export POLL_INTERVAL="5"
export VAD_MODE="low_zh"
export VAD_MODE_JA="low_ja"  # 日语模型独立 VAD 策略，默认与 VAD_MODE 一致

# 是否保留中文 / 日文中间字幕（1/true=保留，默认不保留）
export SAVE_INTERMEDIATE_LRC="true"

# 全流水线开关：开启（true）才执行 ja_zh → large-v3 → LLM 完整流程；
# 关闭（false）则只进行 ja_zh 日中直出，跳过日文转录及 LLM 精调
export FULL_PIPELINE="true"

# 并行转录 worker 数（默认 2，中日文各一个）
export MAX_WORKERS="2"

# LLM 精调（refine）配置
# LLM_MODE 可选值：
#   local  — 使用本地模型（LLM_LOCAL_* 配置，优先）
#   remote — 使用远端 API（LLM_API_* 配置）
#   不设置则向后兼容：有 LLM_API_KEY 走 remote，无 key 走 local
export LLM_MODE="remote"

# 远程 API 配置（LLM_MODE=remote 时使用）
export LLM_API_URL="https://api.deepseek.com/chat/completions"
export LLM_MODEL_NAME="deepseek-v4-flash"
export LLM_API_KEY="sk-xxxxxxxx"  # 远程 API 密钥

# 本地模型配置（LLM_MODE=local 时使用）
# export LLM_LOCAL_API_URL="http://192.168.8.88:1234/v1/chat/completions"
#export LLM_LOCAL_MODEL_NAME="qwen3.5-9b-uncensored-hauhaucs-aggressive-mlx"

export LLM_LOCAL_API_URL="http://192.168.8.88:8000/v1/chat/completions"
export LLM_LOCAL_MODEL_NAME="Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-MLX-mxfp4"
export LLM_LOCAL_API_KEY="omlx"  # 本地模型若需要 API Key 可在此填入

# LLM 精调并发控制：true（默认）— 多批 LLM 调用并行执行（ThreadPoolExecutor）
#                   false          — 多批 LLM 调用串行执行（减少 API 并发压力）
export LLM_CONCURRENT="true"

python worker.py
