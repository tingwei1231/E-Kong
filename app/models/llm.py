"""
app/models/llm.py — LLM 模型管理（llama-cpp-python，GGUF 4-bit）
==================================================================
設計原則：
  - Singleton：全應用共用同一個 Llama 實例，節省 VRAM
  - Lazy init：lifespan 觸發，避免啟動阻塞過長
  - executor：Llama.create_completion 是 CPU/GPU-bound 同步呼叫，
    移入 ThreadPoolExecutor 不阻塞 asyncio event loop
  - 串流（stream=True）預留，本版本先以 non-stream 完成回應

VRAM 佔用（TAIDE-LX-7B-Chat Q4_K_M, n_gpu_layers=35）：~4.8 GB
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from llama_cpp import Llama
from loguru import logger

from app.config import get_settings

settings = get_settings()

# ─── Singleton & 執行緒池 ──────────────────────────────────────────────────────

_model: Llama | None = None
_lock: asyncio.Lock = asyncio.Lock()
# LLM 推論專用 thread pool（大小 = 1，避免並發 VRAM OOM）
_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")


# ─── 生命週期 ─────────────────────────────────────────────────────────────────

async def init_llm() -> None:
    """
    初始化 LLM 模型（應在 FastAPI lifespan 啟動時呼叫）。

    首次呼叫才載入；後續呼叫為 no-op。
    模型檔需事先放至 LLM_MODEL_PATH 指定路徑（建議 Google Drive 掛載）。
    """
    global _model
    async with _lock:
        if _model is not None:
            return

        model_path = settings.llm_model_path
        
        import os
        if not os.path.exists(model_path):
            logger.warning(f"⚠️  找不到本地端模型：{model_path}")
            logger.info("⏳ 系統將自動從 Hugging Face 下載並快取 Qwen2.5-7B 模型...")
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
                    filename="qwen2.5-7b-instruct-q4_k_m.gguf"
                )
                logger.success(f"✅ 模型下載/暫存完成：{model_path}")
            except Exception as e:
                logger.error(f"❌ 自動下載模型失敗：{e}")
                raise

        n_gpu = settings.llm_n_gpu_layers

        logger.info(
            f"🤖 載入 LLM：{model_path}\n"
            f"   GPU layers={n_gpu}  max_tokens={settings.llm_max_tokens}"
        )

        loop = asyncio.get_event_loop()
        _model = await loop.run_in_executor(
            _executor,
            lambda: Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu,     # T4 建議 35；-1 = 全量 offload
                n_ctx=4096,             # context window（7B 模型最大 4096）
                n_batch=512,            # prompt 批次大小
                verbose=False,          # 關閉 llama.cpp 詳細日誌
                use_mmap=True,          # memory-map 大檔，減少記憶體複製
                use_mlock=False,        # Colab 無法 mlock，關閉
            ),
        )
        logger.success("✅ LLM 載入完成。")


def get_llm() -> Llama:
    """取得已初始化的 LLM 實例（未初始化則 raise RuntimeError）。"""
    if _model is None:
        raise RuntimeError("LLM 尚未初始化，請先呼叫 init_llm()")
    return _model


async def close_llm() -> None:
    """釋放 LLM 資源（應用關閉時呼叫）。"""
    global _model
    async with _lock:
        if _model is not None:
            _model = None  # 讓 GC 觸發 __del__ 釋放 VRAM
            _executor.shutdown(wait=False)
            logger.info("🗑️  LLM 已卸載。")


# ─── 推論核心 ─────────────────────────────────────────────────────────────────

async def generate(
    prompt: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
) -> str:
    """
    以給定 prompt 執行 LLM 推論，回傳生成文字。

    Parameters
    ----------
    prompt : str
        已組裝完成的完整 prompt（含系統指令與對話歷史）。
    max_tokens : int | None
        最大生成 token 數；None 則使用 settings.llm_max_tokens。
    temperature : float | None
        溫度；None 則使用 settings.llm_temperature。
    stop : list[str] | None
        停止詞列表，遇到即停止生成。

    Returns
    -------
    str
        LLM 生成的回覆文字（已 strip 首尾空白）。

    Raises
    ------
    RuntimeError
        模型尚未初始化。
    """
    model = get_llm()
    _max_tokens = max_tokens or settings.llm_max_tokens
    _temperature = temperature if temperature is not None else settings.llm_temperature
    _stop = stop or ["</s>", "<|im_end|>", "[/INST]", "User:", "使用者："]

    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(
        _executor,
        lambda: model.create_completion(
            prompt,
            max_tokens=_max_tokens,
            temperature=_temperature,
            stop=_stop,
            echo=False,         # 不回傳 prompt 本身
            stream=False,
        ),
    )

    text: str = output["choices"][0]["text"].strip()
    finish_reason: str = output["choices"][0]["finish_reason"]

    if finish_reason == "length":
        logger.warning("⚠️  LLM 達到 max_tokens 限制，回覆可能截斷。")

    if not text:
        logger.warning(
            f"⚠️  LLM 輸出空字串｜finish={finish_reason}｜"
            f"raw={output['choices'][0]['text']!r}"
        )
    else:
        logger.debug(
            f"🤖 LLM 生成｜tokens={output['usage']['completion_tokens']}｜"
            f"finish={finish_reason}｜{repr(text[:80])}"
        )
    return text
