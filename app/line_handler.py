"""
app/line_handler.py — LINE Messaging API 事件處理器
=====================================================
職責：
  - 接收並驗證 LINE Webhook 簽名
  - 路由各類型事件（TextMessage、AudioMessage、…）
  - Step 1：實作 Echo Bot；後續 Step 替換為 STT/LLM/TTS 管線

設計原則：
  - 每種事件獨立方法，便於後續擴充
  - 回覆使用非同步 httpx，避免阻塞 FastAPI event loop
  - 所有 LINE API 錯誤以 loguru 記錄並妥善包裝
"""

from __future__ import annotations

import hashlib
import hmac
import json
from base64 import b64encode
from typing import Any

import httpx
from fastapi import Header, HTTPException, Request
from loguru import logger

from app.config import get_settings
from app.models.stt import transcribe
from app.services.agent import chat, clear_history, parse_command
from app.services.audio import cleanup_temp_file, process_line_audio
from app.services.tts import try_reply_audio

settings = get_settings()

# LINE Reply API 端點
_LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"

# 非同步 HTTP client（模組層級單例，重用 connection pool）
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """取得（或建立）共用 AsyncClient。"""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=10.0,
            headers={
                "Authorization": f"Bearer {settings.line_channel_access_token}",
                "Content-Type": "application/json",
            },
        )
    return _http_client


async def close_http_client() -> None:
    """應用關閉時優雅釋放 HTTP client 資源。"""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        logger.info("LINE HTTP client 已關閉。")


# ─── 簽名驗證 ─────────────────────────────────────────────────────────────────

def verify_line_signature(body: bytes, signature: str) -> None:
    """
    驗證 LINE Webhook X-Line-Signature。

    Parameters
    ----------
    body : bytes
        原始 request body（未解析）。
    signature : str
        來自 X-Line-Signature header 的 Base64 HMAC-SHA256 值。

    Raises
    ------
    HTTPException
        簽名不符時回傳 400，防止偽造請求。
    """
    channel_secret = settings.line_channel_secret.encode("utf-8")
    computed = hmac.new(channel_secret, body, hashlib.sha256).digest()
    expected = b64encode(computed).decode("utf-8")

    if not hmac.compare_digest(expected, signature):
        logger.warning("❌ LINE 簽名驗證失敗，疑似偽造請求。")
        raise HTTPException(status_code=400, detail="Invalid LINE signature")


# ─── 回覆工具 ─────────────────────────────────────────────────────────────────

async def reply_text(reply_token: str, text: str) -> None:
    """
    使用 Reply API 回覆純文字訊息。

    Parameters
    ----------
    reply_token : str
        LINE 事件中提供的 reply token（僅可使用一次，30 秒過期）。
    text : str
        回覆的文字內容，最長 5000 字元。
    """
    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text[:5000]}],
    }
    client = get_http_client()
    try:
        resp = await client.post(_LINE_REPLY_URL, json=payload)
        resp.raise_for_status()
        logger.debug(f"✉️  已回覆文字訊息（token: {reply_token[:8]}…）")
    except httpx.HTTPStatusError as exc:
        logger.error(
            f"LINE Reply API 錯誤 [{exc.response.status_code}]: {exc.response.text}"
        )


# ─── 事件路由 ─────────────────────────────────────────────────────────────────

async def handle_text_message(event: dict[str, Any]) -> None:
    """
    處理文字訊息事件：指令解析 → 情緒偵測 → LLM 推論 → 回覆。
    """
    reply_token: str = event["replyToken"]
    user_id: str = event["source"].get("userId", "unknown")
    text: str = event["message"]["text"]

    logger.info(f"📩 TextMessage from {user_id}: {text!r}")

    # ── 指令解析 ───────────────────────────────────────────────────────────────
    cmd = parse_command(text)
    if cmd == "reset":
        clear_history(user_id)
        await reply_text(reply_token, "🧹 記憶已清除，我們重新開始吧！😊")
        return

    # ── LLM Agent 推論 ───────────────────────────────────────────────────────────
    try:
        reply, emotion = await chat(user_id, text)
        logger.info(f"✅ LLM│{user_id}│{emotion.zh}{emotion.emoji}│{reply[:60]}")
        # 先嘗試语音回覆，失敗則 fallback 純文字
        voice_ok = await try_reply_audio(reply_token, reply, language=None)
        if not voice_ok:
            await reply_text(reply_token, reply)
    except RuntimeError:
        # LLM 尚未初始化（例：模型檔案不存在）→ 回落 Echo
        logger.warning("⚠️  LLM 未就緒，回落 Echo 模式。")
        await reply_text(reply_token, f"[Ē-Kóng Echo] {text}")
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"文字處理未預期錯誤：{exc}")
        await reply_text(reply_token, "系統暫時異常，請稍後再試。")


async def handle_audio_message(event: dict[str, Any]) -> None:
    """
    處理語音訊息事件：LINE m4a → ffmpeg WAV → Whisper STT → 回覆文字。

    後續 Step 3 會在 STT 之後接 LLM Agent 推論。
    """
    reply_token: str = event["replyToken"]
    message_id: str = event["message"]["id"]
    user_id: str = event["source"].get("userId", "unknown")

    logger.info(f"🎤 AudioMessage id={message_id} from {user_id}")

    wav_path: str | None = None
    try:
        # 1. 下載 LINE 音訊 → 轉換為 16kHz WAV
        wav_path = await process_line_audio(message_id)

        # 2. Whisper STT
        result = await transcribe(wav_path)

        if not result.text:
            await reply_text(reply_token, "訊不清楚，能再說一次嗎？🙏")
            return

        logger.info(
            f"✅ STT｜{user_id}｜語言={result.language}（{result.language_probability:.0%}）｜{result.text!r}"
        )

        # 3. LLM        # STT 完成後進 LLM Agent
        try:
            reply, emotion = await chat(user_id, result.text)
            logger.info(f"✅ LLM│{user_id}│{emotion.zh}{emotion.emoji}│{reply[:60]}")
        except RuntimeError:
            # LLM 尚未初始化 → graceful fallback：回覆 STT 轉錄文字
            lang_label = {"zh": "🇨🇳 中文", "nan": "🇹🇼 台語"}.get(
                result.language, result.language.upper()
            )
            reply = (
                f"🗣️ 我聽到你說（{lang_label}）：\n{result.text}"
            )

        # TTS 语音回覆（依偷測語言自動選擇引擎）
        voice_ok = await try_reply_audio(reply_token, reply, language=result.language)
        if not voice_ok:
            await reply_text(reply_token, reply)

    except FileNotFoundError as exc:
        logger.error(f"音訊檔案不存在：{exc}")
        await reply_text(reply_token, "音訊下載失敗，請稍後再試。")
    except RuntimeError as exc:
        logger.error(f"STT 失敗：{exc}")
        await reply_text(reply_token, "語音辨識失敗，請稍後再試。")
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"音訊處理未預期錯誤：{exc}")
        await reply_text(reply_token, "系統暫時異常，請稍後再試。")
    finally:
        # 確保 WAV 暫存檔被清理
        if wav_path:
            cleanup_temp_file(wav_path)


async def handle_follow_event(event: dict[str, Any]) -> None:
    """處理加好友 / 解除封鎖事件。"""
    reply_token: str = event["replyToken"]
    user_id: str = event["source"].get("userId", "unknown")
    logger.info(f"👋 Follow event from {user_id}")
    await reply_text(
        reply_token,
        "你好！我是 Ē-Kóng (會講)，你的中台語語音陪伴助手。\n"
        "傳訊或傳語音給我，我來陪你聊天！😊",
    )


# ─── 主進入點 ──────────────────────────────────────────────────────────────────

async def dispatch_events(request: Request, x_line_signature: str = Header(...)) -> None:
    """
    Webhook 主處理函式；由 FastAPI router 呼叫。

    流程：
      1. 讀取 body → 驗證簽名
      2. 解析 events 陣列
      3. 依 event type 路由至對應 handler
    """
    body: bytes = await request.body()
    verify_line_signature(body, x_line_signature)

    payload: dict[str, Any] = json.loads(body)
    events: list[dict[str, Any]] = payload.get("events", [])

    logger.debug(f"📨 收到 {len(events)} 個 LINE 事件。")

    for event in events:
        event_type: str = event.get("type", "")
        if event_type == "message":
            msg_type: str = event["message"]["type"]
            if msg_type == "text":
                await handle_text_message(event)
            elif msg_type == "audio":
                await handle_audio_message(event)
            else:
                logger.debug(f"未處理的訊息類型：{msg_type}")
        elif event_type == "follow":
            await handle_follow_event(event)
        elif event_type == "unfollow":
            logger.info(f"🚶 Unfollow: {event['source'].get('userId')}")
        else:
            logger.debug(f"未處理的事件類型：{event_type}")

