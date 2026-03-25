"""
app/models/tts_tw.py — MMS-TTS 台語 TTS 模型管理
==================================================
使用 facebook/mms-tts-nan（Min-Nan / 閩南語）VITS 模型。

VRAM 佔用：~300 MB（VITS 小型模型）

語言代碼說明：
  - "nan" = Min-Nan（閩南語 / 台語）by Whisper/ISO 639-3
  - HuggingFace model: "facebook/mms-tts-nan"

注意：MMS-TTS 輸入需為台羅拼音（Tâi-lô），而非漢字。
因目前缺乏可靠的漢字→台羅轉換器，本版有兩種策略：
  1. 若輸入已是拼音（detect_romanized 判斷）→ 直接合成
  2. 否則 → 轉以中文 ChatTTS 合成（fallback），並記錄警告

後續可接入 g2p_twblg（台語 G2P）提升覆蓋率。
"""

from __future__ import annotations

import asyncio
import io
from functools import lru_cache

import numpy as np
import soundfile as sf
from loguru import logger

# ─── Singleton ────────────────────────────────────────────────────────────────

_processor = None  # VitsTokenizer
_model = None      # VitsModel
_lock: asyncio.Lock = asyncio.Lock()

_MODEL_ID = "facebook/mms-tts-nan"


async def init_tts_tw() -> None:
    """初始化 MMS-TTS 台語模型（應在 FastAPI lifespan 啟動時呼叫）。"""
    global _processor, _model
    async with _lock:
        if _model is not None:
            return
        logger.info(f"🔊 載入 MMS-TTS 台語模型：{_MODEL_ID}")
        try:
            from transformers import VitsModel, VitsTokenizer  # type: ignore
            import torch

            loop = asyncio.get_event_loop()

            def _load():
                device = "cuda" if torch.cuda.is_available() else "cpu"
                proc = VitsTokenizer.from_pretrained(_MODEL_ID)
                mdl = VitsModel.from_pretrained(_MODEL_ID).to(device)
                mdl.eval()
                return proc, mdl

            _processor, _model = await loop.run_in_executor(None, _load)
            logger.success(f"✅ MMS-TTS [{_MODEL_ID}] 載入完成。")
        except ImportError:
            logger.warning("⚠️  transformers 套件未安裝，台語 TTS 不可用。")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ MMS-TTS 載入失敗：{exc}")


def get_tts_tw():
    """取得已初始化的 MMS-TTS 模型（未初始化返回 None）。"""
    return _model


async def close_tts_tw() -> None:
    """釋放 MMS-TTS 資源。"""
    global _processor, _model
    async with _lock:
        if _model is not None:
            _processor = None
            _model = None
            logger.info("🗑️  MMS-TTS 已卸載。")


# ─── 合成核心 ─────────────────────────────────────────────────────────────────

def is_romanized(text: str) -> bool:
    """
    粗略判斷文字是否為台羅拼音（非漢字）。

    MMS-TTS 需要台羅拼音輸入；漢字直接輸入效果極差。
    判斷方法：若漢字比例 < 20% 視為拼音。
    """
    if not text:
        return False
    han_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return (han_count / len(text)) < 0.2


def _do_infer_tw(text: str) -> bytes:
    """
    同步執行 MMS-TTS 推論，回傳 WAV bytes（16kHz mono）。

    在 executor 中呼叫此函式。
    """
    if _model is None or _processor is None:
        raise RuntimeError("台語 TTS 尚未初始化")

    import torch

    inputs = _processor(text=text, return_tensors="pt")
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = _model(**inputs)

    audio_arr = output.waveform[0].squeeze().cpu().numpy()  # shape: (T,)
    sample_rate: int = _model.config.sampling_rate  # 通常 16000

    buf = io.BytesIO()
    sf.write(buf, audio_arr, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


async def synthesize_tw(text: str) -> bytes:
    """
    非同步合成台語 TTS。

    若輸入為漢字（非拼音），記錄警告並嘗試直接合成
    （效果可能較差，建議後續接 G2P）。

    Parameters
    ----------
    text : str
        台羅拼音或台語漢字文字。

    Returns
    -------
    bytes
        WAV 格式音訊（16kHz mono）。

    Raises
    ------
    RuntimeError
        MMS-TTS 未初始化。
    """
    if _model is None:
        raise RuntimeError("台語 TTS 尚未初始化，請先呼叫 init_tts_tw()")

    if not is_romanized(text):
        logger.warning(
            f"⚠️  台語 TTS 輸入含漢字（{text[:30]}…），效果可能不佳。"
            "建議：後續接 g2p_twblg 進行漢字→台羅轉換。"
        )

    loop = asyncio.get_event_loop()
    wav_bytes = await loop.run_in_executor(None, _do_infer_tw, text)
    logger.debug(f"🔉 MMS-TTS 台語合成完成（{len(wav_bytes) / 1024:.1f} KB）")
    return wav_bytes
