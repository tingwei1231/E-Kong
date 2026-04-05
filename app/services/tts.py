"""
app/services/tts.py — TTS 語言路由 & 語音回覆服務
===================================================
職責：
  1. 根據語言代碼選擇合適的 TTS 引擎
  2. 合成語音並上傳至 LINE（Audio Message）
  3. 失敗時回傳 False，由呼叫方決定 fallback 策略

語言路由：
  - "nan"（台語）→ MMS-TTS
  - 其餘 → ChatTTS 中文

設計原則：
  - 不依賴 line_handler（避免循環 import）
  - 只負責「合成 + 上傳」，回覆動作由 line_handler 執行
"""

from __future__ import annotations

from loguru import logger

from app.models.tts_zh import get_tts_zh, synthesize_zh
from app.services.audio import push_audio_message


# ─── 語言路由 ─────────────────────────────────────────────────────────────────

def _pick_engine(language: str | None) -> str:
    """依語言代碼選擇 TTS 引擎（目前全由中文引擎負責）。"""
    return "zh"


async def synthesize(text: str, language: str | None = None) -> bytes | None:
    """
    合成語音，回傳 WAV bytes。失敗或未初始化時回傳 None。

    Parameters
    ----------
    text : str
        要合成的文字。
    language : str | None
        語言代碼（決定 TTS 引擎）；None 使用中文。

    Returns
    -------
    bytes | None
        WAV bytes，或 None（失敗時）。
    """
    engine = _pick_engine(language)
    try:
        # 強制使用中文 TTS
        if get_tts_zh() is not None:
            logger.debug(f"🎤 中文 TTS（ChatTTS）：{text[:30]}…")
            return await synthesize_zh(text)
        else:
            logger.warning("⚠️  TTS 引擎未初始化，跳過語音合成。")
            return None
    except Exception as exc:  # noqa: BLE001
        logger.error(f"❌ TTS 合成失敗：{exc}")
        return None


async def try_push_audio(
    user_id: str,
    text: str,
    language: str | None = None,
) -> bool:
    """
    嘗試合成語音並以 LINE Audio Message 推播。

    Parameters
    ----------
    user_id : str
        LINE 使用者 ID。
    text : str
        LLM 回覆文字（TTS 輸入）。
    language : str | None
        語言代碼（用於選擇 TTS 引擎）。

    Returns
    -------
    bool
        True 表示語音推播成功；False 表示失敗（呼叫方應改用純文字）。
    """
    wav_bytes = await synthesize(text, language)
    if wav_bytes is None:
        return False

    try:
        await push_audio_message(user_id, wav_bytes)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"⚠️  LINE 語音推播失敗：{exc}")
        return False

