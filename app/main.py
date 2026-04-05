"""
app/main.py — FastAPI 應用程式進入點
=====================================
架構：
  - lifespan context manager 管理啟動 / 關閉資源
  - /webhook  : LINE Messaging API Webhook 端點
  - /health   : 健康檢查（ngrok / LINE Dashboard 可用）
  - 全域 Exception handler 確保不洩漏 500 堆疊至 LINE
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger

from app.config import get_settings
from app.line_handler import close_http_client, dispatch_events, verify_line_signature
from app.models.llm import close_llm, init_llm
from app.models.stt import close_stt, init_stt
from app.models.tts_tw import close_tts_tw, init_tts_tw
from app.models.tts_zh import close_tts_zh, init_tts_zh

settings = get_settings()


# ─── Lifespan（啟動 / 關閉鉤子） ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager。
    啟動：載入 Whisper STT 模型。
    關閉：釋放 HTTP client 與 STT 模型資源。
    """
    logger.info("═" * 50)
    logger.info("  Ē-Kóng FastAPI 啟動中...")
    logger.info(f"  LLM 模型路徑 : {settings.llm_model_path}")
    logger.info(f"  Whisper 模型 : {settings.whisper_model_size} ({settings.whisper_compute_type})")
    logger.info("═" * 50)

    # ── STT 初始化 ───────────────────────────────────────────────────────────
    await init_stt()

    # ── LLM 初始化（模型需存在於 LLM_MODEL_PATH 指定路徑）─────────────────────
    try:
        await init_llm()
    except Exception as exc:  # noqa: BLE001
        # 模型檔不存在時不阻斷啟動，Echo Bot 仍可用
        logger.warning(f"⚠️  LLM 載入失敗，系統以 Echo 模式啟動：{exc}")

    # ── TTS 初始化（套件未安裝時不阻斷服務）──────────────────────────────────
    try:
        await init_tts_zh()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"⚠️  ChatTTS 載入失敗，中文語音不可用：{exc}")

    try:
        await init_tts_tw()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"⚠️  MMS-TTS 載入失敗，台語語音不可用：{exc}")

    yield  # ← 應用正常運行期間

    # ── 清理資源 ────────────────────────────────────────────────────────────
    await close_http_client()
    await close_stt()
    await close_llm()
    await close_tts_tw()
    await close_tts_zh()
    logger.info("🛑 Ē-Kóng FastAPI 已正常關閉。")


# ─── FastAPI 實例 ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Project Ē-Kóng (會講)",
    description="情緒感知中台雙向語音陪伴 Agent — LINE Bot Webhook",
    version="0.1.0",
    docs_url="/docs",      # Swagger UI（開發用）
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ─── 全域 Exception Handler ───────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    捕捉所有未處理例外，避免堆疊資訊洩漏給 LINE 伺服器。
    LINE 需要收到 200 OK，否則會重試導致訊息重複。
    """
    logger.exception(f"未預期例外：{exc}")
    return JSONResponse(
        status_code=200,  # 刻意回 200 避免 LINE 重試
        content={"status": "error", "detail": "Internal server error"},
    )


# ─── 路由 ─────────────────────────────────────────────────────────────────────

from fastapi import BackgroundTasks

@app.post("/webhook", status_code=200, summary="LINE Webhook 接收端點")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_line_signature: str = Header(...)
) -> PlainTextResponse:
    """
    LINE Messaging API Webhook 端點。

    流程：
      1. 讀取 body 並同步驗證簽名
      2. 將事件委派給 `dispatch_events` 在背景執行
      3. 立即回傳 200 OK（確保 LINE 1 秒內收到以避免重試斷線死結）
    """
    body: bytes = await request.body()
    verify_line_signature(body, x_line_signature)
    
    background_tasks.add_task(dispatch_events, body)
    return PlainTextResponse("OK")


@app.get("/health", summary="健康檢查")
async def health() -> dict:
    """
    健康檢查端點（增強版）。

    回報：
      - 各模型載入狀態（STT / LLM / TTS 中文 / TTS 台語）
      - GPU VRAM 使用量（若有 CUDA）
      - 服務版本
    """
    from app.models.llm import get_llm
    from app.models.stt import get_stt
    from app.models.tts_tw import get_tts_tw
    from app.models.tts_zh import get_tts_zh

    # 模型狀態
    models: dict[str, str] = {}
    try:
        get_stt()
        models["stt"] = "ok"
    except RuntimeError:
        models["stt"] = "not_loaded"

    try:
        get_llm()
        models["llm"] = "ok"
    except RuntimeError:
        models["llm"] = "not_loaded"

    models["tts_zh"] = "ok" if get_tts_zh() is not None else "not_loaded"
    models["tts_tw"] = "ok" if get_tts_tw() is not None else "not_loaded"

    # GPU VRAM
    gpu: dict[str, str] = {}
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            gpu["device"] = torch.cuda.get_device_name(0)
            gpu["allocated_mb"] = f"{allocated:.0f}"
            gpu["total_mb"] = f"{total:.0f}"
            gpu["usage_pct"] = f"{allocated / total * 100:.1f}%"
        else:
            gpu["device"] = "cpu"
    except ImportError:
        gpu["device"] = "unavailable"

    return {
        "status": "ok",
        "service": "Ē-Kóng",
        "version": app.version,
        "models": models,
        "gpu": gpu,
    }


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"message": "Ē-Kóng (會講) is running. POST /webhook for LINE events."}
