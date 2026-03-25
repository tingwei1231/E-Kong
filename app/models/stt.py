"""
app/models/stt.py — Faster-Whisper STT 模型管理
=================================================
設計原則：
  - Singleton（單例）：模型只載入一次，節省 VRAM 與啟動時間
  - Lazy init：首次呼叫 `get_stt()` 時才載入，FastAPI lifespan 觸發
  - 線程安全：asyncio.Lock 防止並發重複初始化
  - INT8 量化：T4 15GB VRAM 下 base 模型僅佔 ~300MB

VRAM 預算（base, INT8）：~300 MB
"""

from __future__ import annotations

import asyncio
from typing import NamedTuple

from faster_whisper import WhisperModel
from loguru import logger

from app.config import get_settings

settings = get_settings()

# ─── 型別 ──────────────────────────────────────────────────────────────────────

class TranscriptSegment(NamedTuple):
    """單一轉錄片段。"""
    start: float    # 秒
    end: float      # 秒
    text: str       # 轉錄文字（已 strip）


class TranscriptResult(NamedTuple):
    """完整轉錄結果。"""
    text: str                          # 全文（所有段落合併）
    segments: list[TranscriptSegment]  # 詳細時間戳清單
    language: str                      # 偵測語言代碼（zh / nan / …）
    language_probability: float        # 語言信心值 0–1


# ─── Singleton 管理 ───────────────────────────────────────────────────────────

_model: WhisperModel | None = None
_lock: asyncio.Lock = asyncio.Lock()


async def init_stt() -> None:
    """
    初始化 Faster-Whisper 模型（應在 FastAPI lifespan 啟動時呼叫）。

    若已初始化則跳過，確保只載入一次。
    """
    global _model
    async with _lock:
        if _model is not None:
            return
        logger.info(
            f"🎙️  載入 Whisper 模型：{settings.whisper_model_size} "
            f"({settings.whisper_device}, {settings.whisper_compute_type})"
        )
        # faster_whisper.WhisperModel 是同步 CPU-bound，放到 executor 避免阻塞 event loop
        loop = asyncio.get_event_loop()
        _model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                settings.whisper_model_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                # 首次使用會自動下載模型至 ~/.cache/huggingface/hub
                # Colab 重啟後快取消失；可改掛載 Drive 路徑避免重複下載
                download_root=None,
            ),
        )
        logger.success(
            f"✅ Whisper [{settings.whisper_model_size}] 載入完成"
        )


def get_stt() -> WhisperModel:
    """
    取得已初始化的 STT 模型（若未初始化則 raise RuntimeError）。

    在 lifespan 完成後呼叫，永遠不會收到 None。
    """
    if _model is None:
        raise RuntimeError("STT 模型尚未初始化，請先呼叫 init_stt()")
    return _model


async def close_stt() -> None:
    """釋放 STT 模型（應用關閉時呼叫）。"""
    global _model
    async with _lock:
        if _model is not None:
            # WhisperModel 無 explicit close，設 None 讓 GC 回收 VRAM
            _model = None
            logger.info("🗑️  STT 模型已卸載。")


# ─── 轉錄核心 ─────────────────────────────────────────────────────────────────

async def transcribe(audio_path: str) -> TranscriptResult:
    """
    對指定音訊檔進行語音轉文字。

    Parameters
    ----------
    audio_path : str
        本機音訊檔路徑（WAV/M4A/MP3 均支援）。
        建議先轉為 16kHz mono WAV 以獲得最佳速度。

    Returns
    -------
    TranscriptResult
        包含全文、片段時間戳、偵測語言的結果物件。

    Raises
    ------
    RuntimeError
        模型尚未初始化。
    FileNotFoundError
        音訊檔案不存在。
    """
    import os
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音訊檔案不存在：{audio_path}")

    model = get_stt()

    logger.debug(f"🔊 開始轉錄：{audio_path}")

    # WhisperModel.transcribe 是同步 CPU/GPU-bound，移到 executor 執行
    loop = asyncio.get_event_loop()
    segments_iter, info = await loop.run_in_executor(
        None,
        lambda: model.transcribe(
            audio_path,
            language=None,          # 自動偵測語言（中文/台語均可）
            beam_size=5,
            vad_filter=True,        # VAD 過濾靜音，減少幻覺（hallucination）
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
            word_timestamps=False,  # 不需字詞級時間戳，節省計算
        ),
    )

    # 消費 generator（在 executor 內完成，避免跨 thread 問題）
    segments: list[TranscriptSegment] = [
        TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
        )
        for seg in segments_iter
        if seg.text.strip()  # 過濾純空白片段
    ]

    full_text = " ".join(s.text for s in segments)
    logger.info(
        f"✅ 轉錄完成｜語言：{info.language}（{info.language_probability:.0%}）｜"
        f"文字：{full_text[:60]}{'…' if len(full_text) > 60 else ''}"
    )

    return TranscriptResult(
        text=full_text,
        segments=segments,
        language=info.language,
        language_probability=info.language_probability,
    )
