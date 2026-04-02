"""
app/models/tts_zh.py — ChatTTS 中文 TTS 模型管理
==================================================
設計原則：
  - Singleton：全應用共用一個 ChatTTS 實例
  - Lazy init：lifespan 觸發
  - executor：ChatTTS.infer() 是 CPU/GPU-bound 同步呼叫
  - 輸出：WAV bytes（16kHz, mono）供後續上傳至 LINE

VRAM 佔用：~1.5 GB（GPT + DVAE + Vocos）

ChatTTS 特點：
  - 支援音色克隆（spk_emb），本版使用固定 speaker emb 確保一致性
  - 支援韻律控制標籤（[uv_break], [laugh]），適合擬人化回覆
"""

from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

from app.config import get_settings

settings = get_settings()

# ─── Singleton ────────────────────────────────────────────────────────────────

_model = None  # chattts.Chat instance
_lock: asyncio.Lock = asyncio.Lock()


async def init_tts_zh() -> None:
    """初始化 ChatTTS 中文 TTS 模型（應在 FastAPI lifespan 啟動時呼叫）。"""
    global _model
    async with _lock:
        if _model is not None:
            return
        logger.info("🔊 載入 ChatTTS 中文 TTS 模型...")
        try:
            # 解決 PyTorch 2.4+ 移除了 torch.serialization.FILE_LIKE 的相容性問題
            import typing
            import torch
            if not hasattr(torch.serialization, 'FILE_LIKE'):
                torch.serialization.FILE_LIKE = typing.Any

            import ChatTTS  # type: ignore
            loop = asyncio.get_event_loop()

            def _load() -> "ChatTTS.Chat":
                chat = ChatTTS.Chat()
                chat.load(compile=False)  # compile=True 會提速但首次編譯慢
                return chat

            _model = await loop.run_in_executor(None, _load)
            logger.success("✅ ChatTTS 載入完成。")
        except ImportError:
            logger.warning("⚠️  chattts 套件未安裝，中文 TTS 不可用。")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ ChatTTS 載入失敗：{exc}")


def get_tts_zh():
    """取得已初始化的 ChatTTS 實例（未初始化返回 None）。"""
    return _model


async def close_tts_zh() -> None:
    """釋放 ChatTTS 資源。"""
    global _model
    async with _lock:
        if _model is not None:
            _model = None
            logger.info("🗑️  ChatTTS 已卸載。")


# ─── 合成核心 ─────────────────────────────────────────────────────────────────

# 固定 speaker embedding（確保音色一致；可改為 random_speaker 讓每次隨機）
_FIXED_SPK_EMB: np.ndarray | None = None

_MAX_TEXT_LEN = 200  # ChatTTS 建議單次推論文字長度上限


def _get_speaker_emb():
    """懶初始化固定 speaker embedding（首次使用時採樣）。"""
    global _FIXED_SPK_EMB
    if _FIXED_SPK_EMB is None and _model is not None:
        rand_spk = _model.sample_random_speaker()
        _FIXED_SPK_EMB = rand_spk
    return _FIXED_SPK_EMB


def _do_infer(text: str) -> bytes:
    """
    同步執行 ChatTTS 推論，回傳 WAV bytes（24kHz mono）。

    兼容 ChatTTS 0.1.x（Chat.InferCodeParams class attr）
    與 0.2.x+（params 以 dict 傳入 infer()）。
    """
    if _model is None:
        raise RuntimeError("ChatTTS 尚未初始化")

    # 截斷過長文字
    if len(text) > _MAX_TEXT_LEN:
        logger.warning(f"⚠️  文字過長（{len(text)}），截斷至 {_MAX_TEXT_LEN} 字。")
        text = text[:_MAX_TEXT_LEN]

    import ChatTTS  # type: ignore

    spk_emb = _get_speaker_emb()

    # ── 動態偵測 ChatTTS API 版本 ────────────────────────────────────────────
    # 0.1.x：Chat.InferCodeParams / Chat.RefineTextParams 直接可用
    # 0.2.x+：改為 params dict，透過 infer() 的 keyword args 傳入
    try:
        # 嘗試新版 API（0.2.x+）：直接以 infer_code_params dict 傳入
        wavs = _model.infer(
            [text],
            skip_refine_text=False,
            refine_text_only=False,
            params_infer_code={
                "spk_emb": spk_emb,
                "temperature": 0.0003,
                "top_P": 0.7,
                "top_K": 20,
            },
            params_refine_text={
                "prompt": "[oral_2][laugh_0][break_6]",
            },
            use_decoder=True,
        )
    except TypeError:
        # Fallback：舊版 API（0.1.x），params 以 dataclass 傳入
        params_infer = ChatTTS.Chat.InferCodeParams(
            spk_emb=spk_emb,
            temperature=0.0003,
            top_P=0.7,
            top_K=20,
        )
        params_refine = ChatTTS.Chat.RefineTextParams(
            prompt="[oral_2][laugh_0][break_6]",
        )
        wavs = _model.infer(
            [text],
            params_infer_code=params_infer,
            params_refine_text=params_refine,
            use_decoder=True,
            skip_refine_text=False,
        )

    audio_arr = np.squeeze(wavs[0])  # shape: (T,)

    buf = io.BytesIO()
    sf.write(buf, audio_arr, samplerate=24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


async def synthesize_zh(text: str) -> bytes:
    """
    非同步合成中文 TTS，回傳 WAV bytes。

    Parameters
    ----------
    text : str
        要合成的中文文字。

    Returns
    -------
    bytes
        WAV 格式音訊二進位資料（24kHz mono）。

    Raises
    ------
    RuntimeError
        ChatTTS 未初始化。
    """
    if _model is None:
        raise RuntimeError("中文 TTS 尚未初始化，請先呼叫 init_tts_zh()")

    loop = asyncio.get_event_loop()
    wav_bytes = await loop.run_in_executor(None, _do_infer, text)
    logger.debug(f"🔉 ChatTTS 合成完成（{len(wav_bytes) / 1024:.1f} KB）")
    return wav_bytes
