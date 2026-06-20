import gc
import math
import os
import re
import numpy as np

import mlx.core as mx

from .vad import VadOptions, get_speech_timestamps
from .audio import decode_audio

SAMPLING_RATE = 16000

# ── 第二轮转录用的细粒度 VAD 参数 ──
# threshold:      语音概率阈值，>0.5 判定为语音（值越低越灵敏）
# speech_pad_ms:  语音段前后填充的静音毫秒数（避免切掉词头词尾）
# min_speech_duration_ms:  最短语音段，短于此值的噪声段会被丢弃
# min_silence_duration_ms: 两段语音间至少间隔此毫秒才切分为不同段（越大越倾向于合并）
_REFINE_VAD = VadOptions(
    threshold=0.2,
    speech_pad_ms=200,
    min_speech_duration_ms=200,
    min_silence_duration_ms=500,
)


def format_lrc_timestamp(seconds: float) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    just_seconds = math.floor(seconds) % 60
    just_hundredths = math.floor(100 * (seconds - math.floor(seconds)))
    just_minutes = math.floor(seconds / 60)
    return f"[{just_minutes:02d}:{just_seconds:02d}.{just_hundredths:02d}]"




def _split_sentences(text: str, max_chars: int = 30) -> list[str]:
    """将文本拆分为短句。

    优先在 。！？…」 处切分，每句末尾必然带标点。
    超出 max_chars 的句子在 、 处兜底切分。
    """
    parts = re.split(r'(?<=[。！？…」!?])', text)
    result = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) <= max_chars:
            result.append(p)
        else:
            sub = re.split(r'(?<=[、，,])', p)
            for s in sub:
                s = s.strip()
                if s:
                    result.append(s)
    # 合并无意义的单字片段
    cleaned = []
    for item in result:
        if len(item) <= 1 and cleaned:
            cleaned[-1] += item
        else:
            cleaned.append(item)

    # ── 合并连续的短拟声词片段（含小写假名的短假名句） ──
    # 例如「むちゅ。ぬちゅ。ぐちゅ。」→ 一句，避免精炼拆成过多行
    def _is_short_moan(s: str) -> bool:
        ct = re.sub(r'[、。！？…「」,\.!?\s]', '', s)
        if not ct or len(ct) > 4:
            return False
        if re.search(r'[\u4e00-\u9fff]', ct):  # 含汉字 → 不是拟声词
            return False
        return bool(re.search(r'[っゃゅょぁぃぅぇぉゎァィゥェォッャュョヮ]', ct))

    merged_moans = []
    last_was_moan = False
    for item in cleaned:
        is_moan = _is_short_moan(item)
        if merged_moans and is_moan and last_was_moan:
            merged_moans[-1] += item
        else:
            merged_moans.append(item)
        last_was_moan = is_moan

    return merged_moans if merged_moans else [text.strip()]




def _refine_cluster_timestamps(
    audio_path: str,
    cluster_texts: list[str],
    seg_audio_start: float,
    seg_audio_end: float,
    language: str,
    model_path: str,
) -> list[tuple[float, str]]:
    """对密集簇对应的音频段做细粒度转录，返回每句的时间戳和拆分后的短句。

    流程：
    1. 提取簇对应的音频段（含前后余量）
    2. 用细粒度 VAD + word_timestamps=True 重新转录
    3. 将转录得到的文本按句末标点拆成短句
    4. 用词级 midpoint 计算每句的时间戳
    """
    import mlx_whisper
    import soundfile as sf

    # 读取并用细粒度 VAD 分割音频段
    full_wav = decode_audio(audio_path, sampling_rate=SAMPLING_RATE)
    start_sample = max(0, int(seg_audio_start * SAMPLING_RATE))
    end_sample = min(int(seg_audio_end * SAMPLING_RATE), len(full_wav))
    if end_sample <= start_sample:
        del full_wav; gc.collect()
        n = max(len(cluster_texts), 1)
        span = max(seg_audio_end - seg_audio_start, 0.5)
        return [(seg_audio_start + (i / n) * span, cluster_texts[min(i, len(cluster_texts)-1)]) for i in range(n)]
    segment_wav = full_wav[start_sample:end_sample]
    del full_wav; gc.collect()

    # 细粒度 VAD 分割
    vad_segments = get_speech_timestamps(segment_wav, _REFINE_VAD)
    if not vad_segments:
        n = max(len(cluster_texts), 1)
        span = max(seg_audio_end - seg_audio_start, 0.5)
        return [(seg_audio_start + (i / n) * span, cluster_texts[min(i, len(cluster_texts)-1)]) for i in range(n)]

    # 对每个 VAD 段单独转录，每段 2~5s，Whisper 能给出精确词级时间戳
    refined_words: list[tuple[float, str]] = []  # (midpoint, word_text)
    for ci, chunk in enumerate(vad_segments):
        chunk_audio = segment_wav[chunk["start"]:chunk["end"]]
        orig_start = seg_audio_start + chunk["start"] / SAMPLING_RATE
        if len(chunk_audio) < SAMPLING_RATE * 0.2:
            continue  # 太短的段跳过

        temp_path = f"{audio_path}.refine_{os.getpid()}_{ci}.wav"
        try:
            sf.write(temp_path, chunk_audio, SAMPLING_RATE)
            result = mlx_whisper.transcribe(
                temp_path,
                verbose=False,
                language=language,
                path_or_hf_repo=model_path,
                fp16=True,
                no_speech_threshold=0.1,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1,
                word_timestamps=True,
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        for r in result.get("segments", []):
            for w in r.get("words", []):
                wt = w.get("word", "").strip()
                if not wt:
                    continue
                ws = orig_start + w["start"]
                we = orig_start + w["end"]
                refined_words.append(((ws + we) / 2, wt))

    del segment_wav
    if not refined_words:
        n = max(len(cluster_texts), 1)
        span = max(seg_audio_end - seg_audio_start, 0.5)
        return [(seg_audio_start + (i / n) * span, cluster_texts[min(i, len(cluster_texts)-1)]) for i in range(n)]

    # 将第二轮转录的全文按标点拆成短句
    full_text = "".join(w for _, w in refined_words)
    sentences = _split_sentences(full_text)
    if not sentences:
        sentences = [full_text]

    # ── 字符→时间插值对齐：每个字映射到其所属词的 midpoint ──
    # 因为 "".join(sentences) == full_text 严格成立，所以句子的首字符
    # 在 full_text 中的位置精确指向 char_times 中的正确条目，
    # 不存在 gap 累积问题。
    char_times: list[float] = []
    for midpoint, word_text in refined_words:
        char_times.extend([midpoint] * len(word_text))

    result_lines: list[tuple[float, str]] = []
    char_pos = 0
    print(f"    [DEBUG] 第二轮转录: 原文={full_text[:60]}... 拆为{len(sentences)}句")
    for si, sent in enumerate(sentences):
        if char_pos < len(char_times):
            ts = char_times[char_pos]
        else:
            ts = result_lines[-1][0] + 1.0 if result_lines else seg_audio_start
        result_lines.append((ts, sent))
        print(f"    [DEBUG]   句{si}: [{ts:.3f}] {sent[:40]}")
        char_pos += len(sent)

    # ── 如果仍有相同时间戳的句子，合并回长句 ──
    # 精度不够强行散布没有意义，合并后给 LLM 完整上下文
    if len(result_lines) > 1:
        merged: list[tuple[float, str]] = []
        i = 0
        while i < len(result_lines):
            ts_i, text_i = result_lines[i]
            j = i + 1
            fmt_i = format_lrc_timestamp(ts_i)
            while j < len(result_lines) and format_lrc_timestamp(result_lines[j][0]) == fmt_i:
                text_i += result_lines[j][1]
                j += 1
            merged.append((ts_i, text_i))
            i = j
        result_lines = merged

    return result_lines


def get_vad_parameters(mode: str) -> VadOptions:
    if mode == "high":
        return VadOptions(
            threshold=0.5,
            speech_pad_ms=200,
            min_speech_duration_ms=250,
            min_silence_duration_ms=250,
        )
    elif mode == "low":
        return VadOptions(
            threshold=0.45,
            speech_pad_ms=400,
            min_speech_duration_ms=250,
            min_silence_duration_ms=2000,
        )
    elif mode == "low1":
        return VadOptions(
            threshold=0.2,
            speech_pad_ms=400,
            min_speech_duration_ms=250,
            min_silence_duration_ms=3000,
        )
    elif mode == "asmr":
        return VadOptions(
            threshold=0.15,
            speech_pad_ms=500,
            min_speech_duration_ms=250,
            min_silence_duration_ms=5000,
        )
    elif mode == "low_zh":
        return VadOptions(
            threshold=0.2,
            speech_pad_ms=400,
            min_speech_duration_ms=250,
            min_silence_duration_ms=3000,
        )
    elif mode == "low_ja":
        return VadOptions(
            threshold=0.2,
            speech_pad_ms=400,
            min_speech_duration_ms=250,
            min_silence_duration_ms=3000,
        )
    else:
        return None


def transcribe_audio(audio_path: str, model_path: str, language: str = "ja", vad_mode: str = "low") -> str:
    import mlx_whisper

    # 从 model_path 提取简短标识用于日志
    path_lower = model_path.lower()
    if "ja_zh" in path_lower or "zh" in path_lower:
        tag = "[zh]"
    elif "ja" in path_lower:
        tag = "[ja]"
    else:
        tag = "[?]"

    print(f"{tag} 正在读取音频文件: {audio_path}")
    wav = decode_audio(audio_path, sampling_rate=SAMPLING_RATE)
    print(f"{tag} 音频文件读取完成，采样率: {SAMPLING_RATE}, 长度: {len(wav)}")

    print(f"{tag} 正在进行语音活动检测(VAD)，模式: {vad_mode}...")
    vad_params = get_vad_parameters(vad_mode)

    if vad_params is None:
        print(f"{tag} 不使用VAD，将整个音频视为一个语音片段")
        speech_timestamps = [{"start": 0, "end": len(wav)}]
    else:
        speech_timestamps = get_speech_timestamps(wav, vad_params)

    print(f"{tag} VAD 检测到 {len(speech_timestamps)} 个语音片段")

    if not speech_timestamps:
        print(f"{tag} 未检测到语音")
        return ""

    audio_chunks = []
    chunk_start_times = []
    chunk_durations = []

    # ── 仅在 asmr 模式下：片段间插入短静音 ──
    if vad_mode == "asmr":
        gap_samples = int(SAMPLING_RATE * 0.3)
        audio_parts = []
        segment_info = []
        concat_pos = 0.0

        for i, chunk in enumerate(speech_timestamps):
            chunk_audio = wav[chunk["start"]:chunk["end"]]
            duration = (chunk["end"] - chunk["start"]) / SAMPLING_RATE
            orig_start = chunk["start"] / SAMPLING_RATE

            audio_parts.append(chunk_audio)
            segment_info.append((concat_pos, orig_start, duration))
            concat_pos += duration

            if i < len(speech_timestamps) - 1:
                audio_parts.append(np.zeros(gap_samples, dtype=np.float32))
                concat_pos += 0.3

        concat_audio = np.concatenate(audio_parts)
    else:
        for chunk in speech_timestamps:
            start_sample = chunk["start"]
            end_sample = chunk["end"]
            chunk_audio = wav[start_sample:end_sample]
            start_time = start_sample / SAMPLING_RATE
            duration = (end_sample - start_sample) / SAMPLING_RATE
            audio_chunks.append(chunk_audio)
            chunk_start_times.append(start_time)
            chunk_durations.append(duration)

        concat_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)
    print(f"{tag} 音频片段处理完成，总长度: {len(concat_audio)}")

    import soundfile as sf
    temp_audio_path = os.path.join(
        os.path.dirname(audio_path),
        f"_temp_{tag}_{os.path.basename(audio_path)}.wav",
    )
    try:
        sf.write(temp_audio_path, concat_audio, SAMPLING_RATE)
        print(f"{tag} 保存临时音频文件: {temp_audio_path}")

        # ── 第一轮转录（segment 级，获得准确文本）──
        del wav, concat_audio, audio_chunks
        gc.collect()

        print(f"{tag} 第一轮转录 (模型: {model_path})...")
        trans = mlx_whisper.transcribe(
            temp_audio_path,
            verbose=False,
            language=language,
            path_or_hf_repo=model_path,
            fp16=True,
            no_speech_threshold=0.1,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1,
        )
        print(f"  {tag} 转录完成，正在处理时间戳...")
        if not trans.get("segments"):
            print(f"{tag} 未检测到语音内容")
            return ""

        # ── 收集 segment 原始时间戳和文本 ──
        raw_segments: list[tuple[float, float, str]] = []
        for r in trans["segments"]:
            text = r.get("text", "").strip()
            if not text or text == "!":
                continue

            segment_start = r["start"]
            segment_end = r["end"]

            original_start = 0.0
            original_end = 0.0

            if vad_mode == "asmr":
                for concat_pos, orig_start, dur in segment_info:
                    concat_end = concat_pos + dur
                    if segment_start >= concat_pos and segment_start < concat_end:
                        original_start = orig_start + (segment_start - concat_pos)
                    if segment_end >= concat_pos and segment_end <= concat_end:
                        original_end = orig_start + (segment_end - concat_pos)
                    if original_start > 0.0 and original_end > 0.0:
                        break
            else:
                cumulative_duration = 0.0
                for i, duration in enumerate(chunk_durations):
                    if segment_start >= cumulative_duration and segment_start < cumulative_duration + duration:
                        offset_in_chunk = segment_start - cumulative_duration
                        original_start = chunk_start_times[i] + offset_in_chunk
                    if segment_end >= cumulative_duration and segment_end <= cumulative_duration + duration:
                        offset_in_chunk = segment_end - cumulative_duration
                        original_end = chunk_start_times[i] + offset_in_chunk
                    cumulative_duration += duration
                    if original_start > 0 and original_end > 0:
                        break

            if original_end <= original_start:
                original_end = original_start + (segment_end - segment_start)

            raw_segments.append((original_start, original_end, text))

        if not raw_segments:
            print(f"{tag} 未检测到语音内容")
            return ""

        # 先释放第一轮转录结果
        del trans
        gc.collect()

        # ── 按顺序处理每个段：文本长度检测 + 第二轮精炼 ──
        # 对 [ja] 段：如果 _split_sentences 会拆成多句，做第二轮转录获取每句真实时间戳
        # 对 [zh] 段：直接输出，不处理
        final_segments: list[tuple[float, float, str]] = []
        tag_is_ja = (tag == "[ja]")
        for seg_i, (start, end, text) in enumerate(raw_segments):
            if not tag_is_ja:
                final_segments.append((start, end, text))
                continue

            sentences = _split_sentences(text)
            if len(sentences) <= 1:
                final_segments.append((start, end, text))
            else:
                refined = _refine_cluster_timestamps(
                    audio_path, [text],
                    max(0, start - 0.5), end + 0.5,
                    language, model_path,
                )
                print(f"  {tag} 当前段{seg_i+1} 总段{len(raw_segments)} (音频 {start:.1f}s-{end:.1f}s): 1长句 → 拆为{len(refined)}短句")
                for i, (ts, t) in enumerate(refined):
                    seg_end = refined[i + 1][0] if i + 1 < len(refined) else end
                    final_segments.append((ts, seg_end, t))
                # 强制清理 MLX Metal 缓存
                from mlx_whisper.transcribe import ModelHolder
                if ModelHolder.model is not None:
                    try:
                        mx.set_cache_limit(0)
                    except Exception:
                        pass
                    mx.clear_cache()
                    _trigger = mx.zeros((1,))
                    del _trigger
                    mx.set_cache_limit(512 * 1024 ** 2)
                gc.collect()

        # ── 非 [ja] 模型（[zh]）：做完就彻底回收内存 ──
        if tag != "[ja]":
            from mlx_whisper.transcribe import ModelHolder
            if ModelHolder.model is not None:
                ModelHolder.model = None
                ModelHolder.model_path = None
            for _ in range(3):
                try:
                    mx.set_cache_limit(0)
                except Exception:
                    pass
                mx.clear_cache()
                gc.collect()

        # ── 所有段按时间戳排序（第二轮返回的时间戳可能乱序）──
        final_segments.sort(key=lambda x: x[0])

        # ── 合并相邻的重复文本段（如连续的喘气/呻吟短句）──
        merged: list[tuple[float, float, str]] = []
        for seg in final_segments:
            if merged and seg[2] == merged[-1][2]:
                # 文本相同 → 延长前一段的结束时间
                merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]), merged[-1][2])
            else:
                merged.append(seg)
        final_segments = merged

        lrc_lines = [f"{format_lrc_timestamp(ts)} {text}" for ts, _, text in final_segments]

        del raw_segments, chunk_start_times, chunk_durations
        if vad_mode == "asmr":
            del segment_info, audio_parts
        gc.collect()

        return "\n".join(lrc_lines)
    finally:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
