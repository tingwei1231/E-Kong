"""
app/services/audio.py — 音訊下載 & 前處理服務
===============================================
職責：
  1. 從 LINE Content API 下載音訊訊息（m4a 格式）
  2. 將 m4a 轉換為 16kHz mono WAV（ffmpeg）供 Whisper 最佳化使用
  3. 管理暫存檔案的生命週期（避免 Colab 磁碟爆滿）

依賴：
  - ffmpeg（Colab 已預裝）
  - httpx（非同步下載）

設計原則：
  - 所有暫存檔使用 tempfile，確保異常也能清理
  - ffmpeg 轉換在 ProcessPoolExecutor 執行（CPU-bound，釋放 event loop）
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

import httpx
from loguru import logger

from app.config import get_settings

settings = get_settings()

# LINE Content API 端點
_LINE_CONTENT_URL = "https://api-data.line.me/v2/bot/message/{message_id}/content"


# ─── 下載 ─────────────────────────────────────────────────────────────────────

async def download_line_audio(message_id: str) -> Path:
    """
    從 LINE Content API 下載音訊訊息，存為暫存 .m4a 檔案。

    Parameters
    ----------
    message_id : str
        LINE 音訊訊息 ID（來自 event["message"]["id"]）。

    Returns
    -------
    Path
        暫存 .m4a 檔案路徑（呼叫方負責刪除，或使用 process_line_audio 自動清理）。

    Raises
    ------
    httpx.HTTPStatusError
        LINE API 回傳非 200 狀態碼。
    """
    url = _LINE_CONTENT_URL.format(message_id=message_id)
    headers = {
        "Authorization": f"Bearer {settings.line_channel_access_token}",
    }

    logger.debug(f"⬇️  下載音訊 message_id={message_id}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()

        # 建立暫存檔（不自動刪除，由呼叫方或 process_line_audio 管理）
        tmp = tempfile.NamedTemporaryFile(
            suffix=".m4a",
            delete=False,
            dir=tempfile.gettempdir(),
        )
        tmp.write(resp.content)
        tmp.close()

    logger.debug(f"✅ 音訊下載完成：{tmp.name}（{len(resp.content) / 1024:.1f} KB）")
    return Path(tmp.name)


# ─── 轉換 ─────────────────────────────────────────────────────────────────────

def _ffmpeg_convert(input_path: str, output_path: str) -> None:
    """
    使用 ffmpeg 將音訊轉換為 16kHz mono WAV（同步，在 executor 中呼叫）。

    Parameters
    ----------
    input_path : str
        來源音訊路徑（.m4a / .mp3 / …）。
    output_path : str
        目標 WAV 路徑。

    Raises
    ------
    RuntimeError
        ffmpeg 執行失敗（包含 stderr 輸出以便 debug）。
    """
    cmd = [
        "ffmpeg",
        "-y",                    # 覆蓋輸出檔（不詢問）
        "-i", input_path,
        "-ar", "16000",          # 採樣率：16kHz（Whisper 最佳化）
        "-ac", "1",              # 單聲道（mono）
        "-c:a", "pcm_s16le",    # 16-bit PCM WAV
        "-loglevel", "error",   # 只顯示錯誤，減少噪音
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 轉換失敗（code {result.returncode}）：{result.stderr}"
        )


async def convert_to_wav(input_path: Path) -> Path:
    """
    非同步將音訊轉換為 16kHz mono WAV。

    ffmpeg 為 CPU-bound，透過 ProcessPoolExecutor 執行不阻塞 event loop。

    Parameters
    ----------
    input_path : Path
        來源音訊路徑。

    Returns
    -------
    Path
        轉換後的暫存 WAV 路徑。
    """
    output_path = input_path.with_suffix(".wav")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        _ffmpeg_convert,
        str(input_path),
        str(output_path),
    )

    logger.debug(f"🔄 音訊轉換完成：{input_path.name} → {output_path.name}")
    return output_path


# ─── 複合流程 ─────────────────────────────────────────────────────────────────

async def process_line_audio(message_id: str) -> str:
    """
    完整音訊處理管線：下載 m4a → 轉 WAV → 回傳 WAV 路徑。

    暫存的 .m4a 檔案在轉換完成後立即刪除。
    呼叫方應在 STT 完成後刪除 WAV 檔案（使用 cleanup_temp_file）。

    Parameters
    ----------
    message_id : str
        LINE 音訊訊息 ID。

    Returns
    -------
    str
        16kHz mono WAV 暫存檔路徑（呼叫方負責事後清理）。
    """
    m4a_path: Path | None = None
    try:
        # 1. 下載 m4a
        m4a_path = await download_line_audio(message_id)

        # 2. 轉換為 WAV
        wav_path = await convert_to_wav(m4a_path)

        return str(wav_path)

    finally:
        # 確保 m4a 暫存檔被清理（即使發生例外）
        if m4a_path and m4a_path.exists():
            m4a_path.unlink(missing_ok=True)
            logger.debug(f"🗑️  已刪除暫存 m4a：{m4a_path.name}")


def cleanup_temp_file(path: str) -> None:
    """
    刪除暫存音訊檔案（STT 完成後呼叫）。

    Parameters
    ----------
    path : str
        要刪除的暫存檔路徑。
    """
    try:
        Path(path).unlink(missing_ok=True)
        logger.debug(f"🗑️  已刪除暫存 WAV：{Path(path).name}")
    except OSError as exc:
        logger.warning(f"⚠️  刪除暫存檔失敗：{exc}")


# ─── TTS 音訊上傳至 LINE ──────────────────────────────────────────────────────
#
# LINE Audio Message 規格：
#   - 格式：m4a（AAC-LC）
#   - 最大大小：200 MB
#   - 需先上傳至 LINE 取得 uploadId，再以 reply/push API 回傳
#
# 流程：WAV bytes → ffmpeg 轉 m4a → 上傳 LINE Blob API → 取 uploadId

_LINE_BLOB_UPLOAD_URL = "https://api-data.line.me/v2/bot/audiomessage/upload"
_LINE_REPLY_URL_AUDIO = "https://api.line.me/v2/bot/message/reply"


def _wav_bytes_to_m4a(wav_bytes: bytes) -> bytes:
    """
    將 WAV bytes 透過 ffmpeg 轉換為 m4a（AAC-LC）bytes（同步，在 executor 中呼叫）。

    LINE 只接受 m4a/aac 格式的 Audio Message 附件。
    """
    import subprocess

    # 使用 pipe I/O 避免寫入磁碟
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "wav",       # 輸入格式
        "-i", "pipe:0",    # 從 stdin 讀取
        "-c:a", "aac",     # AAC-LC 編碼
        "-b:a", "128k",    # 128 kbps 品質
        "-movflags", "frag_keyframe+empty_moov",  # 允許 pipe output
        "-f", "mp4",
        "pipe:1",          # 輸出至 stdout
    ]
    result = subprocess.run(
        cmd,
        input=wav_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg WAV→m4a 失敗（code {result.returncode}）：{result.stderr.decode()}"
        )
    return result.stdout


async def upload_audio_to_line(m4a_bytes: bytes) -> str:
    """
    上傳 m4a 音訊至 LINE Blob API，回傳 uploadId。

    Parameters
    ----------
    m4a_bytes : bytes
        m4a 格式音訊二進位資料。

    Returns
    -------
    str
        LINE uploadId（用於 Audio Message 的 originalContentUrl 或 uploadId 欄位）。
    """
    headers = {
        "Authorization": f"Bearer {settings.line_channel_access_token}",
        "Content-Type": "audio/mp4",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            _LINE_BLOB_UPLOAD_URL,
            headers=headers,
            content=m4a_bytes,
        )
        resp.raise_for_status()

    upload_id: str = resp.json().get("uploadId", "")
    logger.debug(f"📤 音訊上傳完成，uploadId={upload_id[:16]}…")
    return upload_id


async def reply_audio_message(
    reply_token: str,
    wav_bytes: bytes,
    duration_ms: int = 0,
) -> None:
    """
    將 TTS WAV bytes 轉換為 m4a、上傳至 LINE，並以 Audio Message 回覆。

    Parameters
    ----------
    reply_token : str
        LINE reply token（30 秒內有效、僅可使用一次）。
    wav_bytes : bytes
        TTS 合成的 WAV 音訊二進位資料。
    duration_ms : int
        音訊時長（毫秒）；0 表示自動估算。
    """
    # 1. WAV → m4a（在 executor 執行，不阻塞 event loop）
    loop = asyncio.get_event_loop()
    m4a_bytes = await loop.run_in_executor(None, _wav_bytes_to_m4a, wav_bytes)

    # 2. 估算時長（若未傳入）
    if duration_ms == 0:
        # WAV 大小粗估：bytes / (sample_rate * channels * bit_depth/8)
        # ChatTTS 24kHz, MMS-TTS 16kHz → 取保守值
        duration_ms = max(1000, len(wav_bytes) // 32)

    # 3. 上傳至 LINE
    upload_id = await upload_audio_to_line(m4a_bytes)

    # 4. 回覆 Audio Message
    headers = {
        "Authorization": f"Bearer {settings.line_channel_access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "replyToken": reply_token,
        "messages": [
            {
                "type": "audio",
                "originalContentUrl": f"https://api-data.line.me/v2/bot/message/{upload_id}/content",
                "duration": duration_ms,
            }
        ],
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(_LINE_REPLY_URL_AUDIO, headers=headers, json=payload)

    if resp.status_code == 200:
        logger.info(f"✅ 已回覆語音訊息（uploadId={upload_id[:12]}…）")
    else:
        logger.error(f"❌ LINE Audio Reply 失敗 [{resp.status_code}]: {resp.text}")
