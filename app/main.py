"""
app/main.py — FastAPI 應用程式進入點
=====================================
架構：
  - lifespan context manager 管理啟動 / 關閉資源
  - /webhook  : LINE Messaging API Webhook 端點
  - /health   : 健康檢查（ngrok / LINE Dashboard 可用）
  - 全域 Exception handler 確保不洩漏 500 堆疊至 LINE

重構說明（v2）：
  - 移除 Faster-Whisper STT、ChatTTS TTS 的初始化（改由 Groq API 處理 STT）
  - 移除 TTS lifespan 鉤子，不再載入任何本地語音模型
  - 純文字回覆架構：LLM 推論 → push_text
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import BackgroundTasks, FastAPI, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger

from app.config import get_settings
from app.line_handler import close_http_client, dispatch_events, verify_line_signature
from app.models.llm import close_llm, init_llm

settings = get_settings()


# ─── Lifespan（啟動 / 關閉鉤子） ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager。
    啟動：載入 GGUF LLM 模型。
    關閉：釋放 HTTP client 與 LLM 資源。
    """
    logger.info("═" * 55)
    logger.info("  Ē-Kóng FastAPI 啟動中（v2 文字輸出架構）")
    logger.info(f"  LLM 模型路徑 : {settings.llm_model_path}")
    logger.info(f"  STT 提供者   : Groq whisper-large-v3 (API)")
    logger.info(f"  TTS          : 已停用（純文字輸出）")
    logger.info("═" * 55)

    # ── LLM 初始化 ───────────────────────────────────────────────────────────
    try:
        await init_llm()
    except Exception as exc:  # noqa: BLE001
        # 模型檔不存在時不阻斷啟動，Echo Bot 仍可用
        logger.warning(f"⚠️  LLM 載入失敗，系統以 Echo 模式啟動：{exc}")

    yield  # ← 應用正常運行期間

    # ── 清理資源 ────────────────────────────────────────────────────────────
    await close_http_client()
    await close_llm()
    logger.info("🛑 Ē-Kóng FastAPI 已正常關閉。")


# ─── FastAPI 實例 ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Project Ē-Kóng (會講)",
    description="排球錦標賽語音場控 Agent — LINE Bot Webhook（語音輸入 × 文字輸出）",
    version="2.0.0",
    docs_url="/docs",
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

@app.post("/webhook", status_code=200, summary="LINE Webhook 接收端點")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_line_signature: str = Header(...),
) -> PlainTextResponse:
    """
    LINE Messaging API Webhook 端點。

    流程：
      1. 讀取 body 並同步驗證簽名
      2. 將事件委派給 `dispatch_events` 在背景執行（避免 timeout）
      3. 立即回傳 200 OK（LINE 要求 1 秒內回應）
    """
    body: bytes = await request.body()
    verify_line_signature(body, x_line_signature)

    background_tasks.add_task(dispatch_events, body)
    return PlainTextResponse("OK")


@app.get("/health", summary="健康檢查")
async def health() -> dict:
    """
    健康檢查端點。

    回報：
      - LLM 載入狀態
      - Groq STT API Key 是否已設定
      - GPU VRAM 使用量（若有 CUDA）
      - 服務版本
    """
    import os

    from app.models.llm import get_llm

    # LLM 狀態
    models: dict[str, str] = {}
    try:
        get_llm()
        models["llm"] = "ok"
    except RuntimeError:
        models["llm"] = "not_loaded"

    # Groq STT 狀態（只檢查 key 是否存在，不實際呼叫 API）
    groq_key = os.getenv("GROQ_API_KEY") or settings.groq_api_key
    models["stt_groq"] = "key_set" if groq_key else "missing_key"
    models["tts"]      = "disabled (text-only mode)"

    # GPU VRAM
    gpu: dict[str, str] = {}
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            gpu["device"]       = torch.cuda.get_device_name(0)
            gpu["allocated_mb"] = f"{allocated:.0f}"
            gpu["total_mb"]     = f"{total:.0f}"
            gpu["usage_pct"]    = f"{allocated / total * 100:.1f}%"
        else:
            gpu["device"] = "cpu"
    except ImportError:
        gpu["device"] = "unavailable"

    return {
        "status":  "ok",
        "service": "Ē-Kóng",
        "version": app.version,
        "models":  models,
        "gpu":     gpu,
    }


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"message": "Ē-Kóng (會講) v2 is running. POST /webhook for LINE events."}
