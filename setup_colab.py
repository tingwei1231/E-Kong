"""
setup_colab.py — Project Ē-Kóng (會講) v2 Colab 自動化啟動腳本
================================================================
架構更新（v2）：
  - STT：本地 Faster-Whisper → Groq API (whisper-large-v3)
  - TTS：已移除（純文字回覆）
  - VRAM 需求：大幅降低（僅 LLM，約 4–5 GB）

職責：
  1. 偵測 Colab 環境，選擇性掛載 Google Drive（存放 GGUF 模型）
  2. 安裝 Python 依賴（含 CUDA 版 llama-cpp-python wheel）
  3. 驗證必要環境變數（LINE token、Groq API Key 等）
  4. 啟動 ngrok tunnel 並取得公開 URL
  5. 動態更新 LINE Webhook URL
  6. 啟動 FastAPI 應用 (uvicorn)

使用方式（Colab cell）：
  !python setup_colab.py

或：
  exec(open('setup_colab.py').read())
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from loguru import logger
from pyngrok import conf as ngrok_conf
from pyngrok import ngrok

# ─── 環境變數 ──────────────────────────────────────────────────────────────────
load_dotenv()

NGROK_AUTH_TOKEN: str         = os.environ["NGROK_AUTH_TOKEN"]
LINE_CHANNEL_ACCESS_TOKEN: str = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
GROQ_API_KEY: str             = os.environ.get("GROQ_API_KEY", "")
APP_PORT: int                 = int(os.getenv("APP_PORT", "8000"))
LOG_LEVEL: str                = os.getenv("LOG_LEVEL", "INFO")

# llama-cpp-python CUDA 版本設定（針對 Colab CUDA 12.x + Python 3.10/3.11）
LLAMA_CPP_VERSION   = "0.3.5"
LLAMA_CPP_INDEX_URL = "https://abetlen.github.io/llama-cpp-python/whl/cu122"

# ─── Logger ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)


# ─── 環境偵測 ──────────────────────────────────────────────────────────────────

def is_colab() -> bool:
    """回傳 True 表示目前在 Google Colab 環境中執行。"""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_google_drive() -> None:
    """
    在 Colab 環境掛載 Google Drive。
    GGUF 模型大檔建議放在 Drive，避免每次重啟 Colab 重新下載。
    """
    if not is_colab():
        logger.info("非 Colab 環境，跳過 Drive 掛載。")
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
        logger.success("✅ Google Drive 已掛載至 /content/drive")

        # 將 HuggingFace 快取指向 Drive，避免每次重啟重新下載
        hf_cache_dir = "/content/drive/MyDrive/ekong_models"
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache_dir
        logger.success(f"📂 模型快取路徑：{hf_cache_dir}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"⚠️  Drive 掛載失敗（可手動掛載）：{exc}")


# ─── 環境變數驗證 ──────────────────────────────────────────────────────────────

def check_env_vars() -> bool:
    """
    驗證所有必要的環境變數是否已設定。
    回傳 True 代表全部通過，False 代表有缺少（仍繼續啟動但會有功能缺失）。
    """
    required = {
        "LINE_CHANNEL_ACCESS_TOKEN": os.getenv("LINE_CHANNEL_ACCESS_TOKEN"),
        "LINE_CHANNEL_SECRET":       os.getenv("LINE_CHANNEL_SECRET"),
        "NGROK_AUTH_TOKEN":          os.getenv("NGROK_AUTH_TOKEN"),
        "LLM_MODEL_PATH":            os.getenv("LLM_MODEL_PATH"),
        "GROQ_API_KEY":              os.getenv("GROQ_API_KEY"),  # STT 必須
    }
    all_ok = True
    for key, val in required.items():
        if not val:
            logger.warning(f"⚠️  環境變數未設定：{key}")
            all_ok = False
        else:
            # 只顯示前 8 字元，避免洩漏 token
            preview = val[:8] + "…" if len(val) > 8 else val
            logger.info(f"   ✓ {key} = {preview}")

    # GROQ_API_KEY 特別警告：沒有這個語音功能全掛
    if not os.getenv("GROQ_API_KEY"):
        logger.error("❌ GROQ_API_KEY 未設定！語音訊息將無法辨識。")
        logger.error("   → 前往 https://console.groq.com/ 申請免費 Key（免費 2000 分鐘/天）")

    return all_ok


# ─── 安裝依賴 ──────────────────────────────────────────────────────────────────

def install_requirements() -> None:
    """
    安裝 requirements.txt，並以 CUDA 版 wheel 覆蓋 llama-cpp-python。

    v2 變更：移除 faster-whisper / ChatTTS / transformers 依賴，
    改裝輕量的 groq 套件（<1MB）。
    """
    req_path = Path(__file__).parent / "requirements.txt"
    if not req_path.exists():
        logger.warning("requirements.txt 未找到，跳過安裝。")
        return

    logger.info("📦 安裝依賴套件（首次約 1–2 分鐘）...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "-q", "-r", str(req_path),
    ])

    logger.info(f"⚡ 安裝 CUDA 版 llama-cpp-python=={LLAMA_CPP_VERSION}...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "-q",
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
        f"llama-cpp-python=={LLAMA_CPP_VERSION}",
        "--extra-index-url", LLAMA_CPP_INDEX_URL,
    ])
    logger.success("✅ 所有依賴安裝完成。")


# ─── ngrok ────────────────────────────────────────────────────────────────────

def start_ngrok(port: int) -> str:
    """
    啟動 ngrok HTTP tunnel，回傳 public HTTPS URL。
    自動清理舊殘留連線（解決 Colab 重啟後 ERR_NGROK_334）。
    """
    ngrok_conf.get_default().auth_token = NGROK_AUTH_TOKEN

    # 清除殘留 ngrok 連線
    try:
        for t in ngrok.get_tunnels():
            ngrok.disconnect(t.public_url)
        ngrok.kill()
    except Exception as e:
        logger.debug(f"清理舊 ngrok 狀態：{e}")

    try:
        os.system("pkill -f ngrok")
        time.sleep(1)
    except Exception:
        pass

    # 指定 127.0.0.1 避免 IPv6 connection refused
    tunnel = ngrok.connect(f"127.0.0.1:{port}", bind_tls=True)
    public_url: str = tunnel.public_url.rstrip("/")
    logger.success(f"🌐 ngrok tunnel 已啟動：{public_url}")
    return public_url


# ─── LINE Webhook ─────────────────────────────────────────────────────────────

def update_line_webhook(public_url: str) -> None:
    """透過 LINE Messaging API 更新 Webhook URL 為 ngrok URL。"""
    webhook_url = f"{public_url}/webhook"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=15) as client:
        resp = client.put(
            "https://api.line.me/v2/bot/channel/webhook/endpoint",
            headers=headers,
            json={"endpoint": webhook_url},
        )

    if resp.status_code == 200:
        logger.success(f"✅ LINE Webhook 已更新為：{webhook_url}")
    else:
        logger.error(f"❌ LINE Webhook 更新失敗 [{resp.status_code}]: {resp.text}")
        raise RuntimeError(f"LINE Webhook 更新失敗：{resp.text}")


def verify_line_webhook() -> None:
    """呼叫 LINE API 驗證 Webhook 連線（測試用）。"""
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    with httpx.Client(timeout=15) as client:
        resp = client.post(
            "https://api.line.me/v2/bot/channel/webhook/test",
            headers=headers,
        )
    result = resp.json()
    if result.get("success"):
        logger.success(f"✅ Webhook 驗證通過（延遲 {result.get('timestamp')}）")
    else:
        logger.warning(f"⚠️  Webhook 驗證回應：{result}")


# ─── FastAPI Server ──────────────────────────────────────────────────────────

def start_server(port: int) -> subprocess.Popen:
    """
    以子程序非阻塞方式啟動 uvicorn FastAPI。

    Returns
    -------
    subprocess.Popen
        uvicorn 子程序 handle（可用於後續監控或終止）。
    """
    # 砍掉先前殘留的 uvicorn process
    try:
        os.system("pkill -f uvicorn")
        time.sleep(1)
    except Exception:
        pass

    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--log-level", LOG_LEVEL.lower(),
    ]
    logger.info(f"🚀 啟動 FastAPI：{' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
    time.sleep(4)  # 等待 server 啟動 + LLM 載入初始化
    if proc.poll() is not None:
        raise RuntimeError("uvicorn 啟動失敗，請檢查日誌。")
    logger.success(f"✅ FastAPI 運行中（PID {proc.pid}）")
    return proc


# ─── 快速冒煙測試 ─────────────────────────────────────────────────────────────

def smoke_test(public_url: str) -> None:
    """
    對 /health 端點發送 GET 請求，驗證服務實際可對外回應。
    這是確認「ngrok → uvicorn → FastAPI」鏈路暢通的最快方式。
    """
    health_url = f"{public_url}/health"
    logger.info(f"🔍 Smoke test：{health_url}")
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(health_url)
        data = resp.json()
        if data.get("status") == "ok":
            logger.success(f"✅ /health OK｜LLM={data['models'].get('llm')}｜STT={data['models'].get('stt_groq')}")
        else:
            logger.warning(f"⚠️  /health 回應異常：{data}")
    except Exception as exc:
        logger.error(f"❌ Smoke test 失敗：{exc}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main() -> None:
    """完整 Colab 環境初始化與啟動流程。"""
    logger.info("═" * 60)
    logger.info("  Project Ē-Kóng (會講) v2 — Colab 環境初始化")
    logger.info(f"  Colab 環境：{'是' if is_colab() else '否（本機開發模式）'}")
    logger.info("═" * 60)

    # Step 0: 掛載 Google Drive（存放 GGUF 模型）
    mount_google_drive()

    # Step 1: 驗證環境變數
    logger.info("🔑 驗證環境變數...")
    check_env_vars()

    # Step 2: 安裝依賴
    install_requirements()

    # Step 3: 啟動 ngrok
    public_url = start_ngrok(APP_PORT)

    # Step 4: 更新 LINE Webhook
    update_line_webhook(public_url)

    # Step 5: 啟動 FastAPI Server
    server_proc = start_server(APP_PORT)

    # Step 6: Webhook 連線驗證
    time.sleep(2)
    verify_line_webhook()

    # Step 7: 冒煙測試（確認 /health 可對外回應）
    smoke_test(public_url)

    logger.info("═" * 60)
    logger.success("🎉 E-Kong v2 系統已就緒！等待 LINE 訊息中...")
    logger.info(f"   Public URL  : {public_url}")
    logger.info(f"   Webhook     : {public_url}/webhook")
    logger.info(f"   Health      : {public_url}/health")
    logger.info(f"   Swagger UI  : {public_url}/docs")
    logger.info("═" * 60)
    logger.info("💡 測試方式：")
    logger.info("   1. 傳語音訊息：Groq Whisper STT → Regex 路由 → LLM → 文字推播")
    logger.info("   2. 傳文字訊息：直接 Regex 路由 → LLM → 文字推播")
    logger.info("   3. 訊息含「改/刪/修改比分」→ 自動擋回，不過 LLM")
    logger.info("═" * 60)

    # 保持主程序存活
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        logger.info("🛑 收到中斷信號，關閉服務...")
        server_proc.terminate()
        ngrok.kill()


if __name__ == "__main__":
    main()
