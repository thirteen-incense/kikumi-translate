import gc
import json
import os
import sys
import time
import threading
import requests

import mlx.core as mx

from server.kikumi_api import KikumiAPI
from transcribe.mlx_whisper import transcribe_audio
from transcribe.refine import refine_lrc

CACHE_FILE = "task_cache.json"


def load_cache() -> dict | None:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return None


def save_cache(task: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(task, f, indent=2)


def clear_cache():
    if os.path.exists(CACHE_FILE):
        os.unlink(CACHE_FILE)


def clear_mlx_memory():
    """彻底清理 MLX Metal 内存池 + Python GC"""
    try:
        mx.set_cache_limit(0)
    except Exception:
        pass
    mx.clear_cache()
    _trigger = mx.zeros((1,))
    del _trigger
    gc.collect()
    mx.set_cache_limit(512 * 1024 ** 2)


def _transcribe_worker(audio_path: str, model_path: str, language: str, vad_mode: str) -> str:
    """在子进程中运行转录（供 ProcessPoolExecutor 使用）。"""
    from transcribe.mlx_whisper import transcribe_audio
    return transcribe_audio(audio_path, model_path, language, vad_mode)


def process_and_finish(api: KikumiAPI, task: dict, audio_path: str, local_audio_dir: str):
    task_id = task["id"]
    secret = task["secret"]
    heartbeat_interval = int(os.environ.get("HEARTBEAT_INTERVAL", "3"))

    api.update_status(task_id, secret, "transcribing")

    stop_heartbeat = threading.Event()
    def heartbeat_loop():
        heartbeat_ok = True
        while not stop_heartbeat.is_set():
            ok = api.heartbeat()
            if not ok:
                if heartbeat_ok:
                    print("  心跳失败")
                heartbeat_ok = False
            else:
                if not heartbeat_ok:
                    print("  心跳恢复")
                heartbeat_ok = True
            stop_heartbeat.wait(heartbeat_interval)
    print("  启动心跳线程")
    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()

    lrc_cache_path = None
    saved_lrc_paths: list[str] = []
    try:
        model_path_zh = task.get("model_path", os.environ.get("MLX_WHISPER_MODEL_PATH", "cache/mlx_models/ja_zh_v2_0.2"))
        model_path_ja = os.environ.get("MLX_WHISPER_MODEL_PATH_JA", "cache/mlx_models/whisper-large-v3-mlx-convert")
        language = task.get("language", os.environ.get("TRANSCRIBE_LANGUAGE", "ja"))
        vad_mode_zh = os.environ.get("VAD_MODE", "low1")
        vad_mode_ja = os.environ.get("VAD_MODE_JA", vad_mode_zh)

        audio_name = task.get("audio_name", os.path.basename(audio_path))
        base, _ = os.path.splitext(audio_name)
        lrc_name = base + ".lrc"

        full_pipeline = os.environ.get("FULL_PIPELINE", "true").lower() in ("1", "true", "yes")
        save_intermediate = os.environ.get("SAVE_INTERMEDIATE_LRC", "").lower() in ("1", "true", "yes")

        # ── 检查 cache/debug 中是否已有转录好的中日文本 ──
        cached_zh_path = None
        cached_ja_path = None
        if save_intermediate and full_pipeline:
            debug_dir = os.path.join(os.path.dirname(local_audio_dir.rstrip("/")), "debug")
            cached_zh_path = os.path.join(debug_dir, f"{base}.zh.lrc")
            cached_ja_path = os.path.join(debug_dir, f"{base}.ja.lrc")

        if save_intermediate and full_pipeline and os.path.exists(cached_zh_path) and os.path.exists(cached_ja_path):
            print(f"cache/debug 中已有转录好的中日字幕，跳过转录直接进行 LLM 精炼")
            with open(cached_zh_path, "r", encoding="utf-8") as f:
                zh_lrc = f.read()
            with open(cached_ja_path, "r", encoding="utf-8") as f:
                ja_lrc = f.read()
        else:
            if full_pipeline:
                # ── 并行转录中日文 ──
                print(f"并行转录中文 (模型: {model_path_zh}) 和日文 (模型: {model_path_ja})...")
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor
                ctx = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=int(os.environ.get("MAX_WORKERS", "2")), mp_context=ctx) as executor:
                    fut_zh = executor.submit(_transcribe_worker, audio_path, model_path_zh, language, vad_mode_zh)
                    fut_ja = executor.submit(_transcribe_worker, audio_path, model_path_ja, language, vad_mode_ja)
                    zh_lrc = fut_zh.result()
                    ja_lrc = fut_ja.result()
                print("  中日转录均已完成")

                # ── 按需保存中间结果 ──
                if save_intermediate:
                    debug_dir = os.path.join(os.path.dirname(local_audio_dir.rstrip("/")), "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    for suffix, content in (("zh", zh_lrc), ("ja", ja_lrc)):
                        path = os.path.join(debug_dir, f"{base}.{suffix}.lrc")
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(content)
                        print(f"debug 中间结果已保存: {path}")
            else:
                print(f"转录中文 (模型: {model_path_zh})...")
                zh_lrc = transcribe_audio(audio_path, model_path_zh, language, vad_mode_zh)
                print("FULL_PIPELINE=false: 跳过日语转录及 LLM 精炼，直接使用日中直出结果")
                ja_lrc = ""

        # ── LLM 润色：以日文 LRC 时间戳为基准，参考中文直译
        if full_pipeline:
            metadata = {
                k: task.get(k)
                for k in ("tags", "vas", "circle")
                if task.get(k)
            }

            # 直接传入 LRC 格式（含时间戳），LLM 保留日文时间戳，优化文本内容
            print(f"  LLM 以日文时间戳为基准进行润色...")
            lrc_result = refine_lrc(zh_lrc, ja_lrc, metadata or None)
            print(f"  LLM 润色完成: {len(lrc_result)} 字符")

            # ── 按需保存润色后 LRC ──
            if save_intermediate:
                debug_dir = os.path.join(os.path.dirname(local_audio_dir.rstrip("/")), "debug")
                os.makedirs(debug_dir, exist_ok=True)
                refined_path = os.path.join(debug_dir, f"{base}.refined.lrc")
                with open(refined_path, "w", encoding="utf-8") as f:
                    f.write(lrc_result)
                print(f"  LLM 润色后 LRC 已保存: {refined_path}")

            # ── 释放中间变量 ──
            del zh_lrc, ja_lrc
            gc.collect()
        else:
            # FULL_PIPELINE=false: zh_lrc 已经是最终结果
            lrc_result = zh_lrc

        print(f"转录完成，字幕文件: {lrc_name}，{len(lrc_result)} 字符")

        lrc_cache_path = os.path.join(local_audio_dir, lrc_name)
        with open(lrc_cache_path, "w", encoding="utf-8") as f:
            f.write(lrc_result)
        saved_lrc_paths.append(lrc_cache_path)
        print(f"字幕已缓存到本地: {lrc_cache_path}")

        while True:
            try:
                result = api.finish(task_id, secret, lrc_result)
                print(f"提交结果到 kikumi: {result}")
                break
            except requests.exceptions.ConnectionError:
                print(f"  提交失败（服务器未就绪），5 秒后重试...")
                time.sleep(5)

        del lrc_result
        gc.collect()

        print(f"任务 {task_id} 完成")
    except Exception as e:
        print(f"转录失败: {e}")
        import traceback
        traceback.print_exc()
        api.update_status(task_id, secret, f"error: {e}")
    finally:
        stop_heartbeat.set()
        hb_thread.join(timeout=10)
        if hb_thread.is_alive():
            print("  心跳线程未及时退出（无害，下一轮会替换）")
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            print(f"已删除本地音频: {audio_path}")
        for p in saved_lrc_paths:
            if os.path.exists(p):
                os.unlink(p)
                print(f"已删除本地字幕缓存: {p}")
        clear_cache()
        print("已清除任务缓存")
        clear_mlx_memory()
        print("  任务完成，MLX Metal 内存与 Python GC 已彻底回收")


def main():
    kikumi_url = os.environ.get("KIKUMI_URL", "http://192.168.10.5:4000")
    worker_name = os.environ.get("WORKER_NAME", "mac_01")
    kikumi_username = os.environ.get("KIKUMI_USERNAME", "admin")
    kikumi_password = os.environ.get("KIKUMI_PASSWORD", "admin")
    model_path = os.environ.get("MLX_WHISPER_MODEL_PATH", "cache/mlx_models/ja_zh_v2_0.2")
    local_audio_dir = os.environ.get("LOCAL_AUDIO_DIR", "./cache/input")
    language = os.environ.get("TRANSCRIBE_LANGUAGE", "ja")
    poll_interval = int(os.environ.get("POLL_INTERVAL", "5"))

    api = KikumiAPI(kikumi_url, worker_name, kikumi_username, kikumi_password)

    print(f"等待 kikumi 服务器: {kikumi_url}")
    while True:
        try:
            if api.register():
                print("注册成功")
                break
        except requests.exceptions.ConnectionError:
            pass
        print(f"  服务器未就绪，{poll_interval} 秒后重试...")
        time.sleep(poll_interval)

    # ── 设置 MLX 内存上限 ──
    try:
        prev_limit = mx.set_memory_limit(16 * 1024 ** 3)  # 16 GB
        print(f"MLX 内存上限已设置: 16 GB (之前: {prev_limit})")
    except Exception as e:
        print(f"MLX 内存上限设置失败（非致命）: {e}")

    print(f"模型路径: {model_path}")
    print(f"kikumi URL: {kikumi_url}")
    print(f"语言: {language}")

    # ── 打印 LLM 配置状态 ──
    full_pipeline = os.environ.get("FULL_PIPELINE", "true").lower() in ("1", "true", "yes")
    if not full_pipeline:
        print("LLM: 未启用（FULL_PIPELINE=false，仅日中直译）")
    else:
        llm_mode = os.environ.get("LLM_MODE") or "auto"
        if llm_mode == "local" or (llm_mode == "auto" and not os.environ.get("LLM_API_KEY")):
            llm_url = os.environ.get("LLM_LOCAL_API_URL", "http://localhost:11434/v1/chat/completions")
            llm_model = os.environ.get("LLM_LOCAL_MODEL_NAME", "qwen2.5:7b")
            print(f"LLM: 已启用 | 模式=local | {llm_model} @ {llm_url}")
        else:
            llm_url = os.environ.get("LLM_API_URL", "https://api.deepseek.com/chat/completions")
            llm_model = os.environ.get("LLM_MODEL_NAME", "deepseek-chat")
            print(f"LLM: 已启用 | 模式=remote | {llm_model} @ {llm_url}")

    resume_unfinished_task(api, local_audio_dir)

    print("开始轮询任务...")
    was_disconnected = False
    while True:
        try:
            task = api.acquire()
            if was_disconnected:
                print("已重新连接")
                was_disconnected = False
        except (requests.exceptions.ConnectionError, requests.exceptions.JSONDecodeError):
            was_disconnected = True
            print(f"  服务器异常，{poll_interval} 秒后重试...")
            time.sleep(poll_interval)
            continue

        if not task:
            time.sleep(poll_interval)
            continue

        print(f"获取到任务: id={task['id']}, audio={task['audio_name']}")

        audio_path = api.download_audio(
            task["audio_url"], task["audio_name"], local_audio_dir
        )
        print(f"音频下载到: {audio_path}")
        print(f"音频文件大小: {os.path.getsize(audio_path)} bytes")

        task["audio_path"] = audio_path
        task["model_path"] = model_path
        task["language"] = language
        save_cache(task)

        process_and_finish(api, task, audio_path, local_audio_dir)

        del task, audio_path
        clear_mlx_memory()
        print("循环内存已回收")


def resume_unfinished_task(api: KikumiAPI, local_audio_dir: str):
    cache = load_cache()
    if not cache:
        print("没有未完成的遗留任务")
        return

    print(f"发现未完成的任务: {cache}")
    audio_path = cache.get("audio_path", "")
    if audio_path and os.path.exists(audio_path):
        print(f"继续处理未完成的任务: {cache.get('id')}")
        process_and_finish(api, cache, audio_path, local_audio_dir)
    else:
        print("本地音频文件不存在，跳过遗留任务")
        clear_cache()


if __name__ == "__main__":
    main()
