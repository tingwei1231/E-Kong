"""
app/line_handler.py — LINE Messaging API 事件處理器（重構版）
=============================================================
重構重點：
  1. STT：本地 Whisper → Groq API (whisper-large-v3)，異步非阻塞
  2. TTS 全部移除：不再有 AudioSendMessage，回應一律為 TextSendMessage
  3. 意圖：不再從 line_handler 呼叫 Intent，由 agent.py 的 Regex 路由內部處理
  4. 回覆策略：先用 reply_token 發等待訊息，完成後用 push_text 推播結果

設計原則：
  - 每種事件獨立方法，便於後續擴充
  - 回覆使用非同步 httpx，避免阻塞 FastAPI event loop
  - Groq STT 在 asyncio.to_thread 中執行（Groq SDK 為同步），釋放 event loop
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import tempfile
from base64 import b64encode
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException
from loguru import logger

from app.config import get_settings
from app.services.agent import AgentResponse, Intent, chat, parse_command
from app.services.audio import cleanup_temp_file, process_line_audio

settings = get_settings()

# LINE Reply API & Push API 端點
_LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
_LINE_PUSH_URL  = "https://api.line.me/v2/bot/message/push"

# 非同步 HTTP client（模組層級單例，重用 connection pool）
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """取得（或建立）共用 AsyncClient。"""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=30.0,
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


# ─── 回覆工具（純文字）────────────────────────────────────────────────────────

async def reply_text(reply_token: str, text: str) -> None:
    """使用 Reply API 回覆純文字（reply_token 只能用一次）。"""
    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text[:5000]}],
    }
    client = get_http_client()
    try:
        resp = await client.post(_LINE_REPLY_URL, json=payload)
        resp.raise_for_status()
        logger.debug(f"✉️  reply_text 成功（token: {reply_token[:8]}…）")
    except httpx.HTTPStatusError as exc:
        logger.error(
            f"LINE Reply API 錯誤 [{exc.response.status_code}]: {exc.response.text}"
        )


async def push_text(user_id: str, text: str) -> None:
    """使用 Push API 主動推播純文字（無時間限制）。"""
    payload = {
        "to": user_id,
        "messages": [{"type": "text", "text": text[:5000]}],
    }
    client = get_http_client()
    try:
        resp = await client.post(_LINE_PUSH_URL, json=payload)
        resp.raise_for_status()
        logger.debug(f"✉️  push_text 成功（to: {user_id[:8]}…）")
    except httpx.HTTPStatusError as exc:
        logger.error(
            f"LINE Push API 錯誤 [{exc.response.status_code}]: {exc.response.text}"
        )


# ─── Groq STT ─────────────────────────────────────────────────────────────────

def _transcribe_with_groq(wav_path: str) -> str:
    """
    同步呼叫 Groq Whisper API 進行語音辨識。

    此函式在 asyncio.to_thread 中執行，不阻塞 event loop。
    Groq 的限速：Audio 25MB/min，免費方案 2000 min/day。

    Parameters
    ----------
    wav_path : str
        16kHz mono WAV 暫存檔路徑。

    Returns
    -------
    str
        辨識出的文字（可能為空字串）。

    Raises
    ------
    RuntimeError
        Groq API 呼叫失敗。
    """
    try:
        from groq import Groq  # 延遲 import，避免未安裝時炸啟動

        api_key = os.getenv("GROQ_API_KEY") or getattr(settings, "groq_api_key", None)
        if not api_key:
            raise RuntimeError("GROQ_API_KEY 未設定，請在 .env 加入此環境變數。")

        client = Groq(api_key=api_key)
        with open(wav_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                response_format="text",
                language="zh",  # 繁體中文優先；Groq 實際上自動偵測效果也不錯
            )
        # response_format="text" 時，transcription 直接是字串
        text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
        logger.info(f"✅ Groq STT 完成｜{text!r:.80}")
        return text

    except ImportError:
        raise RuntimeError("groq 套件未安裝，請執行：pip install groq")
    except Exception as exc:
        logger.error(f"❌ Groq STT 失敗：{exc}")
        raise RuntimeError(f"Groq STT 失敗：{exc}") from exc


async def transcribe_audio(wav_path: str) -> str:
    """
    非同步包裝：在 asyncio.to_thread 執行 Groq STT，不阻塞 event loop。

    Returns
    -------
    str
        辨識文字（空字串代表靜音或辨識失敗）。
    """
    return await asyncio.to_thread(_transcribe_with_groq, wav_path)


# ─── 事件路由 ─────────────────────────────────────────────────────────────────

async def handle_text_message(event: dict[str, Any]) -> None:
    """
    文字訊息事件：
      指令解析 → Regex 路由（agent 內部）→ 單輪 LLM → push_text 回覆
    """
    reply_token: str = event["replyToken"]
    user_id: str = event["source"].get("userId", "unknown")
    text: str = event["message"]["text"]

    logger.info(f"📩 TextMessage from {user_id}: {text!r}")

    # 立刻消耗 reply_token，告知使用者已收到
    await reply_text(
        reply_token,
        f"⏳ 收到「{text[:15]}」，正在處理中，稍後推播結果給您！",
    )

    # ── 指令解析（reset 等特殊指令）──────────────────────────────────────
    cmd = parse_command(text)
    if cmd == "reset":
        await push_text(user_id, "🧹 好的，已清除記憶，重新開始！")
        return

    # ── Agent 推論 ────────────────────────────────────────────────────────
    try:
        resp: AgentResponse = await chat(user_id, text)
        logger.info(f"✅ Agent｜{user_id}｜intent={resp.intent.value}｜{resp.response_text[:60]}")
        await push_text(user_id, resp.response_text)

    except RuntimeError:
        logger.warning("⚠️  LLM 未就緒，回落 Echo 模式。")
        await push_text(user_id, f"[Echo] {text}")
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"文字處理未預期錯誤：{exc}")
        await push_text(user_id, "系統暫時異常，請稍後再試。")


async def handle_audio_message(event: dict[str, Any]) -> None:
    """
    語音訊息事件：
      LINE m4a → ffmpeg WAV → Groq Whisper STT → Regex 路由 → 單輪 LLM → push_text 回覆

    全程純文字回傳，不使用 TTS 或 AudioSendMessage。
    """
    reply_token: str = event["replyToken"]
    message_id: str  = event["message"]["id"]
    user_id: str     = event["source"].get("userId", "unknown")

    logger.info(f"🎤 AudioMessage id={message_id} from {user_id}")

    # 立刻消耗 reply_token，發等待訊息（語音處理較慢）
    await reply_text(
        reply_token,
        "⏳ 收到語音，正在辨識中...稍等一下，辨識完就推播給你！",
    )

    wav_path: str | None = None
    try:
        # 1. 下載 LINE 音訊 → 轉換為 16kHz WAV
        wav_path = await process_line_audio(message_id)

        # 2. Groq Whisper STT（非同步，不阻塞 event loop）
        user_text = await transcribe_audio(wav_path)

        if not user_text:
            await push_text(user_id, "訊號不清楚，能再說一次嗎？🙏")
            return

        logger.info(f"🗣️  STT 結果｜{user_id}｜{user_text!r}")

        # 3. Agent 推論（STT 結果直接餵入）
        resp: AgentResponse = await chat(user_id, user_text)
        logger.info(
            f"✅ Agent｜{user_id}｜intent={resp.intent.value}｜{resp.response_text[:60]}"
        )

        # 4. 純文字推播（附上 STT 辨識結果，方便確認）
        reply_msg = f"🗣️ 我聽到：「{user_text}」\n\n{resp.response_text}"
        await push_text(user_id, reply_msg)

    except RuntimeError as exc:
        logger.error(f"STT/LLM 失敗：{exc}")
        await push_text(user_id, f"處理失敗：{exc}\n請稍後再試或改用文字輸入。")
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"音訊處理未預期錯誤：{exc}")
        await push_text(user_id, "系統暫時異常，請稍後再試。")
    finally:
        # 確保 WAV 暫存檔被清理
        if wav_path:
            cleanup_temp_file(wav_path)


async def handle_follow_event(event: dict[str, Any]) -> None:
    """處理加好友 / 解除封鎖事件。"""
    reply_token: str = event["replyToken"]
    user_id: str     = event["source"].get("userId", "unknown")
    logger.info(f"👋 Follow event from {user_id}")
    await reply_text(
        reply_token,
        "嗨！我是賽場助理 🏐\n"
        "你可以透過語音或文字問我：\n"
        "・「A組第一場現在比分多少？」\n"
        "・「報名截止時間是幾點？」\n"
        "・「廣播：尋找108號選手」\n"
        "（比分修改請洽紀錄台工作人員）",
    )


# ─── 主進入點 ──────────────────────────────────────────────────────────────────

async def dispatch_events(body: bytes) -> None:
    """
    Webhook 背景處理函式。

    流程：
      1. 解析 events 陣列
      2. 依 event type 路由至對應 handler
    """
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
