"""
setup_colab.py — Project Ē-Kóng (會講) Colab 自動化啟動腳本
================================================================
職責：
  1. 偵測 Colab 環境，選擇性掛載 Google Drive
  2. 安裝 Python 依賴（含 CUDA 版 llama-cpp-python wheel）
  3. 啟動 ngrok tunnel 並取得公開 URL
  4. 透過 LINE Messaging API 動態更新 Webhook URL
  5. 啟動 FastAPI 應用 (uvicorn)

使用方式（Colab cell）：
  !python setup_colab.py

或以 subprocess 在背景啟動 FastAPI：
  exec(open('setup_colab.py').read())
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# ── 第三方（在 install_requirements() 呼叫前已確保安裝） ────────────────────────
import httpx
from dotenv import load_dotenv
from loguru import logger
from pyngrok import conf as ngrok_conf
from pyngrok import ngrok

# ─── 環境變數 ──────────────────────────────────────────────────────────────────
load_dotenv()

NGROK_AUTH_TOKEN: str = os.environ["NGROK_AUTH_TOKEN"]
LINE_CHANNEL_ACCESS_TOKEN: str = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# llama-cpp-python 版本 (針對 Colab CUDA 12.2+ 與 Python 3.12)
LLAMA_CPP_VERSION = "0.3.5"
LLAMA_CPP_INDEX_URL = "https://abetlen.github.io/llama-cpp-python/whl/cu124"

# ─── Logger ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL, colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


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
    模型大檔（GGUF）建議放在 Drive，避免每次重啟 Colab 重新下載。
    """
    if not is_colab():
        logger.info("非 Colab 環境，跳過 Drive 掛載。")
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
        logger.success("✅ Google Drive 已掛載至 /content/drive")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"⚠️  Drive 掛載失敗（可手動掛載）：{exc}")


def install_requirements() -> None:
    """
    安裝 requirements.txt 並以 CUDA 預編譯 wheel 覆蓋 llama-cpp-python。

    Colab 已內建 torch/torchaudio；requirements.txt 使用
    `torch ; extra == 'local'` 條件避免 Colab 重裝衝突。
    """
    req_path = Path(__file__).parent / "requirements.txt"
    if not req_path.exists():
        logger.warning("requirements.txt 未找到，跳過安裝。")
        return

    logger.info("📦 安裝依賴套件（可能需要 3–5 分鐘）...")
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


def start_ngrok(port: int) -> str:
    """
    啟動 ngrok HTTP tunnel，回傳 public HTTPS URL。

    Parameters
    ----------
    port : int
        本地 FastAPI 監聽埠號。

    Returns
    -------
    str
        ngrok 提供的 HTTPS public URL（不帶尾斜線）。
    """
    ngrok_conf.get_default().auth_token = NGROK_AUTH_TOKEN

    # 確保清除之前殘留的 ngrok 連線，避免 "already online" (ERR_NGROK_334) 錯誤
    try:
        for t in ngrok.get_tunnels():
            ngrok.disconnect(t.public_url)
        ngrok.kill()
    except Exception as e:
        logger.debug(f"清理舊 ngrok Pyngrok 狀態時略過：{e}")
        
    # 強制透過 OS 層級砍掉背景所有殘留的 ngrok 執行緒 (適用於 Colab 環境重啟 kernel)
    try:
        os.system("pkill -f ngrok")
        time.sleep(1)  # 稍微等待系統釋放 port 與連線
    except Exception as e:
        logger.debug(f"OS 砍掉 ngrok 時發生錯誤: {e}")

    # 指定 127.0.0.1 避免 ngrok 預設解析到 IPv6 [::1] 導致 connection refused
    tunnel = ngrok.connect(f"127.0.0.1:{port}", bind_tls=True)
    public_url: str = tunnel.public_url.rstrip("/")
    logger.success(f"🌐 ngrok tunnel 已啟動：{public_url}")
    return public_url


def update_line_webhook(public_url: str) -> None:
    """
    透過 LINE Messaging API 將 Webhook URL 更新為 ngrok URL。

    Parameters
    ----------
    public_url : str
        ngrok 的 HTTPS public URL。
    """
    webhook_url = f"{public_url}/webhook"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"endpoint": webhook_url}

    with httpx.Client(timeout=15) as client:
        resp = client.put(
            "https://api.line.me/v2/bot/channel/webhook/endpoint",
            headers=headers,
            json=payload,
        )

    if resp.status_code == 200:
        logger.success(f"✅ LINE Webhook 已更新為：{webhook_url}")
    else:
        logger.error(
            f"❌ LINE Webhook 更新失敗 [{resp.status_code}]: {resp.text}"
        )
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


def start_server(port: int) -> subprocess.Popen:
    """
    以子程序非阻塞方式啟動 uvicorn FastAPI 應用。

    Parameters
    ----------
    port : int
        FastAPI 監聽埠號。

    Returns
    -------
    subprocess.Popen
        uvicorn 子程序 handle（可用於後續監控或終止）。
    """
    # 確保強制砍掉先前的 uvicorn process
    try:
        os.system("pkill -f uvicorn")
        time.sleep(1)
    except Exception as e:
        logger.debug(f"清理舊 uvicorn 時略過：{e}")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--log-level", LOG_LEVEL.lower(),
        "--reload",      # Colab 開發階段開啟熱重載
    ]
    logger.info(f"🚀 啟動 FastAPI：{' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
    time.sleep(3)  # 等待 server 啟動完成
    if proc.poll() is not None:
        raise RuntimeError("uvicorn 啟動失敗，請檢查日誌。")
    logger.success(f"✅ FastAPI 運行中（PID {proc.pid}）")
    return proc


def main() -> None:
    """主要執行流程。"""
    logger.info("═" * 60)
    logger.info("  Project Ē-Kóng (會講) — Colab 環境初始化")
    logger.info(f"  Colab 環境：{'是' if is_colab() else '否（本機開發模式）'}")
    logger.info("═" * 60)

    # Step 0: 掛載 Google Drive（存放 GGUF 模型）
    mount_google_drive()

    # Step 1: 安裝依賴
    install_requirements()

    # Step 2: 啟動 ngrok
    public_url = start_ngrok(APP_PORT)

    # Step 3: 更新 LINE Webhook
    update_line_webhook(public_url)

    # Step 4: 啟動 FastAPI Server
    server_proc = start_server(APP_PORT)

    # Step 5: 驗證 Webhook 連線
    time.sleep(2)
    verify_line_webhook()

    logger.info("═" * 60)
    logger.success("🎉 E-Kong 系統已就緒！等待 LINE 訊息中...")
    logger.info(f"   Public URL : {public_url}")
    logger.info(f"   Webhook    : {public_url}/webhook")
    logger.info(f"   Health     : {public_url}/health")
    logger.info("═" * 60)

    # 保持主程序存活（Colab 不會自動鎖定子程序）
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        logger.info("🛑 收到中斷信號，關閉服務...")
        server_proc.terminate()
        ngrok.kill()


if __name__ == "__main__":
    main()
