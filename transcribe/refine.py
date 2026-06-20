"""LLM 精调模块：以日文 LRC 时间戳为基准，参考中文直译润色中文字幕。

时间戳在传给 LLM 前替换为序号，减少 token 消耗，返回后校验行数。

依赖环境变量:
    LLM_API_KEY          — 可选，填入后自动切换为远程 API 模式（DeepSeek / OpenAI 等）
    LLM_API_URL          — 远程 API 端点 (默认: https://api.deepseek.com/v1/chat/completions)
    LLM_MODEL_NAME       — 远程模型名称 (默认: deepseek-chat)
    LLM_LOCAL_API_URL    — 本地模型 API 端点 (默认: http://localhost:11434/v1/chat/completions)
    LLM_LOCAL_MODEL_NAME — 本地模型名称 (默认: qwen2.5:7b)

自动选择逻辑：填入 LLM_API_KEY 时走远程 API，否则走本地模型。
"""

import os
import random
import re
import time
import concurrent.futures
import requests

from requests.exceptions import RequestException

# ── 全局复用 Session ──
_LLM_SESSION = requests.Session()

# ── 加载 system prompt（每次调用时动态读取，修改后即时生效）──
_PROMPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REFINE_PROMPT_PATH = os.path.join(_PROMPT_DIR, "refine-prompt.md")


def _load_system_prompt() -> str:
    with open(_REFINE_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _lrc_to_indexed(lrc: str, start_idx: int = 1) -> tuple[str, dict[int, str]]:
    """将 LRC 格式的时间戳替换为序号，返回 (indexed_text, {index: timestamp})。

    "[00:44.00] 内容" → "{start_idx} 内容"，同时记录 {start_idx: "[00:44.00]"}。
    无法解析的行保留原始内容（不分配序号）。
    """
    lines = lrc.strip().split("\n")
    ts_map: dict[int, str] = {}
    indexed_lines: list[str] = []
    idx = start_idx
    for line in lines:
        m = re.match(r'^(\[\d{2}:\d{2}\.\d{2}\])\s*(.*)', line)
        if m:
            ts_map[idx] = m.group(1)
            indexed_lines.append(f"{idx} {m.group(2).strip()}")
            idx += 1
        else:
            indexed_lines.append(f"{idx} {line.strip()}")
            idx += 1
    return "\n".join(indexed_lines), ts_map


def _parse_json_output(text: str, ts_map: dict[int, str], expected_lines: int) -> str | None:
    """将 LLM 返回的 JSON 解析为 LRC 格式。

    期望结构: {"lines": [{"idx": 1, "text": "..."}, ...]}
    返回 LRC 文本或 None（非合法 JSON 时）。
    """
    import json

    text = text.strip()
    # 去掉可能的 markdown 包裹
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    if not text:
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    lines_data = data.get("lines")
    if not isinstance(lines_data, list):
        return None

    result: list[str] = []
    for item in lines_data:
        if not isinstance(item, dict):
            continue
        idx = item.get("idx")
        txt = item.get("text", "").strip()
        if isinstance(idx, int) and idx in ts_map:
            result.append(f"{ts_map[idx]} {txt}")

    if not result:
        return None

    # 不足的行填空
    used_indices = set()
    for line in result:
        m = re.match(r'^(\[\d{2}:\d{2}\.\d{2}\])\s*(.*)', line)
        if m:
            for idx, ts in ts_map.items():
                if ts == m.group(1):
                    used_indices.add(idx)
                    break
    for idx in range(min(ts_map), max(ts_map) + 1):
        if idx not in used_indices:
            result.append(f"{ts_map[idx]} ")

    # 排序
    def _key(line: str) -> float:
        m = re.match(r'\[(\d{2}):(\d{2})\.(\d{2})\]', line)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100
        return 0.0

    result.sort(key=_key)
    return "\n".join(result[:len(ts_map)])


def _indexed_to_lrc(text: str, ts_map: dict[int, str], expected_lines: int) -> str:
    """将 LLM 返回的序号文本还原为 LRC。

    宽松策略：LLM 输出多少行就用多少行，不足的行填空，多余的行丢弃。
    每行格式需为 `序号 文本`，序号用于映射回时间戳。
    """
    lines = text.strip().split("\n")
    result: list[str] = []

    for line in lines:
        m = re.match(r'^(\d+)\s+(.*)', line)
        if m:
            idx = int(m.group(1))
            content = m.group(2).strip()
            if idx in ts_map:
                result.append(f"{ts_map[idx]} {content}")

    # 不足的行填空
    if len(result) < expected_lines:
        used_indices = set()
        for line in result:
            m = re.match(r'^(\[\d{2}:\d{2}\.\d{2}\])\s*(.*)', line)
            if m:
                # 反向找索引
                for idx, ts in ts_map.items():
                    if ts == m.group(1):
                        used_indices.add(idx)
                        break
        for idx in range(min(ts_map), max(ts_map) + 1):
            if idx not in used_indices:
                result.append(f"{ts_map[idx]} ")

    # 排序
    def _ts_sort_key(line: str) -> float:
        m = re.match(r'\[(\d{2}):(\d{2})\.(\d{2})\]', line)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100
        return 0.0

    result.sort(key=_ts_sort_key)

    return "\n".join(result[:expected_lines])


def _try_parse_as_lrc(text: str, ts_map: dict[int, str], expected_lines: int) -> str | None:
    """尝试按传统 LRC 格式 [MM:SS.hh] 文本 解析，匹配到对应时间戳则还原。

    用于兼容输出传统 LRC 格式而非序号格式的模型。
    """
    lines = text.strip().split("\n")
    result: list[str] = []

    # 建立 timestamp → index 的反向映射
    ts_to_idx = {ts: idx for idx, ts in ts_map.items()}

    for line in lines:
        m = re.match(r'^(\[\d{2}:\d{2}\.\d{2}\])\s*(.*)', line)
        if m:
            ts = m.group(1)
            content = m.group(2).strip()
            if ts in ts_to_idx:
                result.append(f"{ts} {content}")

    if not result:
        return None

    # 按时间戳排序
    def _key(l: str) -> float:
        m = re.match(r'\[(\d{2}):(\d{2})\.(\d{2})\]', l)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100
        return 0.0

    result.sort(key=_key)
    return "\n".join(result[:len(ts_to_idx)])


def _merge_lrc_duplicates(lrc: str) -> str:
    """合并 LRC 中相邻且有相同文本内容的行。

    [00:01.00] 嗯    [00:01.00] 嗯
    [00:01.50] 嗯  →  [00:04.00] 哈
    [00:04.00] 哈
    """
    lines = lrc.strip().split("\n")
    parsed: list[tuple[str, str]] = []  # (timestamp, text)
    for line in lines:
        m = re.match(r'^(\[\d{2}:\d{2}\.\d{2}\])\s*(.*)', line)
        if m:
            parsed.append((m.group(1), m.group(2).strip()))

    if not parsed:
        return lrc

    merged: list[tuple[str, str]] = []
    for ts, text in parsed:
        if merged and text == merged[-1][1]:
            continue  # 丢弃，保留前一个的起始时间戳
        else:
            merged.append((ts, text))

    return "\n".join(f"{ts} {text}" for ts, text in merged)


def _filter_ellipsis_lines(lrc: str) -> str:
    """去除纯……的无效行，下一行内容上移占用其时间戳。

    [00:11.00] ……      →  删除
    [00:14.00] 正文1    →  [00:11.00] 正文1
    [00:16.00] 正文2    →  [00:16.00] 正文2
    """
    lines = lrc.strip().split("\n")
    parsed: list[tuple[str, str, str]] = []  # (raw_line, timestamp, text)
    for line in lines:
        m = re.match(r'^(\[\d{2}:\d{2}\.\d{2}\])\s*(.*)', line)
        if m:
            parsed.append((line, m.group(1), m.group(2).strip()))

    # 过滤纯 …… 行，下一行内容上移
    result: list[str] = []
    pending_ts: str | None = None  # 被删除行中最旧的时间戳
    for raw_line, ts, text in parsed:
        if re.match(r'^…+$', text):
            if pending_ts is None:
                pending_ts = ts  # 只保留第一个（最早）的时间戳
            continue
        if pending_ts:
            result.append(f"{pending_ts} {text}")
            pending_ts = None
        else:
            result.append(f"{ts} {text}")
    # 末尾多余的 …… 直接丢弃

    return "\n".join(result)


BATCH_SIZE = 100
CONTEXT_SIZE = 50


def _validate_lrc_indices(lrc: str, expected_count: int, label: str) -> None:
    """校验 LRC 行数、无重复时间戳、时序递增。"""
    lines = lrc.strip().split("\n")
    timestamps = []
    for line in lines:
        m = re.match(r'^\[(\d{2}):(\d{2})\.(\d{2})', line)
        if m:
            secs = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100
            timestamps.append(secs)
        else:
            # LRC 格式无法解析的行
            if line.strip():
                raise ValueError(f"{label}存在非 LRC 格式行: {line[:40]}")

    if len(timestamps) != expected_count:
        raise ValueError(
            f"{label}行数校验失败: 预期{expected_count}行, 实际{len(timestamps)}行"
        )

    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            raise ValueError(
                f"{label}时间戳乱序: 第{i+1}行({timestamps[i]:.2f}s) < "
                f"第{i}行({timestamps[i-1]:.2f}s)"
            )
        if timestamps[i] == timestamps[i - 1]:
            raise ValueError(
                f"{label}重复时间戳: 第{i+1}行和第{i}行均为{timestamps[i]:.2f}s"
            )


def _call_llm_once(
    user_content: str, ja_ts_map: dict[int, str],
    api_url: str, model_name: str, api_key: str | None,
    is_last_batch: bool,
    batch_label: str = "",
) -> str:
    """单次 LLM 调用（含重试），返回还原后的 LRC 文本。"""
    max_retries = 5
    expected_lines = len(ja_ts_map)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    best_result: tuple[int, str] = (0, "")
    for attempt in range(1, max_retries + 1):
        resp_body = ""
        try:
            messages = [
                {"role": "system", "content": _load_system_prompt()},
                {"role": "user", "content": user_content},
            ]
            estimated_tokens = int(len(user_content) * 2.5)
            dynamic_max_tokens = min(max(5 * estimated_tokens, 64000), 384000)

            resp = _LLM_SESSION.post(
                api_url,
                headers=headers,
                json={
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": dynamic_max_tokens,
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": "high",
                    "response_format": {"type": "json_object"},
                },
                timeout=180,
            )
            try:
                resp.raise_for_status()
                result = resp.json()
            finally:
                resp_body = resp.text
                resp.close()

            choices = result.get("choices", [])
            if not choices:
                raise ValueError(f"LLM 返回异常: 无 choices — {result}")

            refined = choices[0]["message"]["content"].strip()

            refined = refined.strip()
            if not refined:
                raise ValueError("LLM 返回空内容")

            restored = _parse_json_output(refined, ja_ts_map, expected_lines)
            if restored:
                filled_count = expected_lines
                print(f"    解析为 JSON 格式")
            else:
                restored = _indexed_to_lrc(refined, ja_ts_map, expected_lines)
                restored_lines = restored.strip().split("\n")
                filled_count = sum(1 for l in restored_lines if re.match(r'^\[\d{2}:\d{2}\.\d{2}\]\s+\S', l))

                if filled_count < expected_lines // 2:
                    lrc_fallback = _try_parse_as_lrc(refined, ja_ts_map, expected_lines)
                    if lrc_fallback:
                        print(f"    检测到传统 LRC 格式，已转换")
                        restored = lrc_fallback
                        filled_count = expected_lines

                if filled_count == 0:
                    real_lines = [l for l in refined.split("\n") if l.strip()]
                    filled_count = len(real_lines)

            if filled_count > best_result[0]:
                best_result = (filled_count, restored)

            if filled_count == expected_lines:
                if attempt > 1:
                    print(f"    调用成功 (第 {attempt} 次重试)")
                # 校验本批行数/无重复/无乱序
                _validate_lrc_indices(restored, expected_lines,
                                      f"第{batch_label}批" if batch_label else "本批")
                print(f"    行数校验通过 ({expected_lines}行)")
                return restored

            if attempt < max_retries:
                print(f"    行数校验失败 (预期{expected_lines}行, 实际{filled_count}行), 第{attempt+1}次重试...")

        except RequestException as e:
            if resp_body:
                print(f"    >> 服务器响应: {resp_body[:500]}")
            if attempt < max_retries:
                delay = min(2 ** attempt + random.uniform(0, 1), 30)
                print(f"    LLM 调用失败 (第 {attempt}/{max_retries} 次): {e}")
                print(f"    等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
            else:
                print(f"    LLM 调用全部 {max_retries} 次重试均失败")
                raise
        except ValueError as e:
            if resp_body:
                print(f"    >> 服务器响应: {resp_body[:500]}")
            if attempt < max_retries:
                delay = min(2 ** attempt + random.uniform(0, 1), 30)
                print(f"    LLM 返回格式异常 (第 {attempt}/{max_retries} 次): {e}")
                print(f"    等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
            else:
                print(f"    LLM 返回格式异常 (第 {attempt}/{max_retries} 次): {e}")
                raise

    raise RuntimeError(
        f"LLM 行数校验失败: 预期{expected_lines}行，{max_retries}次重试均未匹配"
        f"（最佳结果{best_result[0]}行）"
    )


def _build_batch_prompt(
    ja_batch_lines: list[str],
    context_lines: list[str],
    zh_indexed: str,
    metadata: dict | None,
    batch_idx: int,
    total_batches: int,
    start_idx: int,
    full_count: int,
) -> str:
    """为一批构建 user message。

    前文（context_lines）和本批正文（ja_batch_lines）合并为连续的序号，
    LLM 需输出全部行，后续再截取本批部分。
    """
    all_lines = context_lines + ja_batch_lines
    context_count = len(context_lines)
    all_start = start_idx - context_count
    all_end = start_idx + len(ja_batch_lines) - 1
    all_indexed, _ = _lrc_to_indexed("\n".join(all_lines), start_idx=all_start)

    parts = []
    if metadata:
        meta_lines = []
        if metadata.get("tags"):
            meta_lines.append(f"标签: {', '.join(metadata['tags'])}")
        if metadata.get("vas"):
            meta_lines.append(f"声优: {', '.join(metadata['vas'])}")
        if metadata.get("circle"):
            meta_lines.append(f"社团: {metadata['circle']}")
        if meta_lines:
            parts.append("【作品元数据】")
            parts.extend(meta_lines)
            parts.append("")

    # 前文 + 本批合并为一个连续的日文 LRC
    parts.append(f"【日文 LRC】")
    parts.append(all_indexed)
    parts.append("")

    # 全文 zh 参考
    parts.append("【中文直译 LRC-全文参考】")
    parts.append(zh_indexed)
    parts.append("")

    # 批次说明
    parts.append(f"---")
    parts.append(f"这是第 {batch_idx}/{total_batches} 批处理。")
    parts.append(f"注意：输出的行序需与日文 LRC 完全一致。")
    parts.append(f"本批需输出 {full_count} 行，idx {all_start}-{all_end}。")
    if batch_idx == total_batches:
        parts.append(f"本批是最后一批，注意检查末尾幻觉。")

    return "\n".join(parts)


def refine_lrc(zh_lrc: str, ja_lrc: str, metadata: dict | None = None) -> str:
    """调用 LLM 以日文 LRC 时间戳为基准，参考中文直译润色中文字幕。

    时间戳在传给 LLM 前替换为序号（减少 token 消耗），
    返回后校验行数，不一致则重试。

    当 ja_lrc 行数超过 BATCH_SIZE 时，自动分批并行处理。
    """
    llm_mode = os.environ.get("LLM_MODE") or "auto"
    llm_concurrent = os.environ.get("LLM_CONCURRENT", "true").strip().lower() in ("true", "1", "yes")

    # 决定 URL 和 model（local vs remote）
    if llm_mode == "local":
        api_url = os.environ.get(
            "LLM_LOCAL_API_URL",
            "http://localhost:11434/v1/chat/completions",
        )
        model_name = os.environ.get("LLM_LOCAL_MODEL_NAME", "qwen2.5:7b")
        api_key = os.environ.get("LLM_LOCAL_API_KEY") or None
    else:  # remote / auto
        api_url = os.environ.get(
            "LLM_API_URL",
            "https://api.deepseek.com/v1/chat/completions",
        )
        model_name = os.environ.get("LLM_MODEL_NAME", "deepseek-v4-flash")
        api_key = os.environ.get("LLM_API_KEY") or None

    # ── 过滤纯 …… 无效行（下一行内容上移占用其时间戳）──
    ja_lrc = _filter_ellipsis_lines(ja_lrc)
    zh_lrc = _filter_ellipsis_lines(zh_lrc)

    # ── zh 全量序号化 ──
    zh_indexed, _ = _lrc_to_indexed(zh_lrc)

    # ── 判断是否需要分批 ──
    ja_lines = ja_lrc.strip().split("\n")
    ja_line_count = len(ja_lines)

    if ja_line_count <= BATCH_SIZE:
        # 单批，保持原有逻辑
        ja_indexed, ja_ts_map = _lrc_to_indexed(ja_lrc)
        expected_lines = len(ja_ts_map)

        parts = []
        if metadata:
            meta_lines = []
            if metadata.get("tags"):
                meta_lines.append(f"标签: {', '.join(metadata['tags'])}")
            if metadata.get("vas"):
                meta_lines.append(f"声优: {', '.join(metadata['vas'])}")
            if metadata.get("circle"):
                meta_lines.append(f"社团: {metadata['circle']}")
            if meta_lines:
                parts.append("【作品元数据】")
                parts.extend(meta_lines)
                parts.append("")

        parts.append("【日文 LRC】（行数、行序以此为准，序号保持不动）")
        parts.append(ja_indexed)
        parts.append("")
        parts.append("【中文 LRC】（文本内容参考资料，不使用其行结构）")
        parts.append(zh_indexed)

        user_content = "\n".join(parts)

        print(f"正在调用 LLM ({api_url}, model={model_name}) 进行字幕润色...")
        total_kb = (len(_load_system_prompt()) + len(user_content)) / 1024
        print(f"  日文 LRC: {len(ja_lrc)} 字符 → 序号化后 {len(ja_indexed)} 字符")
        print(f"  中文 LRC: {len(zh_lrc)} 字符 → 序号化后 {len(zh_indexed)} 字符")
        print(f"  发送文本: {total_kb:.1f} KB ({expected_lines} 行)")

        result = _call_llm_once(user_content, ja_ts_map, api_url, model_name, api_key, is_last_batch=True, batch_label="")
        restored_kb = len(result.encode("utf-8")) / 1024
        print(f"  润色完成: {len(result)} 字符，{restored_kb:.1f} KB")
        return result

    # ── 多批处理 ──
    # 构建分批信息
    batches_info = []
    for batch_start in range(0, ja_line_count, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, ja_line_count)
        context_lines = ja_lines[max(0, batch_start - CONTEXT_SIZE):batch_start]
        batch_lines = ja_lines[batch_start:batch_end]
        batches_info.append({
            "context": context_lines,
            "batch": batch_lines,
            "start_idx": batch_start + 1,
            "count": len(batch_lines),
        })

    total_batches = len(batches_info)
    mode_str = "并行" if llm_concurrent else "串行"
    print(f"ja LRC {ja_line_count} 行，拆为 {total_batches} 批{mode_str}处理（每批 {BATCH_SIZE} 行）...")

    # 构建各批的 ts_map（用于 LLM 返回映射）
    batch_results: list[tuple[int, str]] = []  # (start_idx, restored_lrc)

    def _process(bi: int, info: dict) -> tuple[int, str]:
        """处理一批。"""
        batch_idx = bi + 1
        is_last = (batch_idx == total_batches)

        user_content = _build_batch_prompt(
            ja_batch_lines=info["batch"],
            context_lines=info["context"],
            zh_indexed=zh_indexed,
            metadata=metadata,
            batch_idx=batch_idx,
            total_batches=total_batches,
            start_idx=info["start_idx"],
            full_count=len(info["context"]) + info["count"],
        )

        # 构建完整 ts_map（前文+本批），LLM 需输出全部行
        context_count = len(info["context"])
        all_lines = info["context"] + info["batch"]
        all_start = info["start_idx"] - context_count
        _, full_ts_map = _lrc_to_indexed("\n".join(all_lines), start_idx=all_start)
        full_count = len(full_ts_map)
        batch_count = info["count"]

        kb = len(user_content.encode("utf-8")) / 1024
        print(f"  第{batch_idx}/{total_batches}批: idx {all_start}-{all_start+full_count-1} (发送{full_count}行/{kb:.1f}KB) 发送中...")

        full_lrc = _call_llm_once(user_content, full_ts_map, api_url, model_name, api_key, is_last_batch=is_last, batch_label=f"{batch_idx}/{total_batches}")

        # 截取本批部分（跳过前文重叠）：只保留 idx >= start_idx 的行
        batch_tss = {ts for idx, ts in full_ts_map.items() if idx >= info["start_idx"]}
        trimmed = "\n".join(
            line for line in full_lrc.strip().split("\n")
            if any(line.startswith(ts) for ts in batch_tss)
        )

        # 校验截取后行数
        trimmed_count = sum(1 for line in trimmed.strip().split("\n") if line.strip())
        if trimmed_count != batch_count:
            raise ValueError(
                f"第{batch_idx}/{total_batches}批截取失败: "
                f"预期{batch_count}行(idx {info['start_idx']}+), 实际{trimmed_count}行"
            )

        trimmed_kb = len(trimmed.encode("utf-8")) / 1024
        print(f"  第{batch_idx}/{total_batches}批: 完成 ({len(trimmed)}字符/{trimmed_kb:.1f}KB, {batch_count}行)")
        return (info["start_idx"], trimmed)

    if llm_concurrent:
        # 并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_batches) as executor:
            futures = {
                executor.submit(_process, bi, info): bi
                for bi, info in enumerate(batches_info)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results.append(future.result())
                except Exception as e:
                    raise RuntimeError(
                        f"第 {futures[future] + 1}/{total_batches} 批处理失败"
                    ) from e
    else:
        # 串行执行
        for bi, info in enumerate(batches_info):
            batch_results.append(_process(bi, info))

    # 按 start_idx 排序拼接
    batch_results.sort(key=lambda r: r[0])
    all_lrc = "\n".join(r[1] for r in batch_results)

    # ── 最终校验：总行数、无重复、无乱序 ──
    _validate_lrc_indices(all_lrc, ja_line_count, "合并后全文")
    print(f"  最终校验通过: {ja_line_count}行, 无重复/乱序")

    total_kb = len(all_lrc.encode("utf-8")) / 1024
    print(f"  全部 {total_batches} 批完成: {len(all_lrc)} 字符，{total_kb:.1f} KB")
    return all_lrc