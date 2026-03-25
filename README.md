# Project Ē-Kóng (會講) — 開發 Walkthrough

## 完整目錄結構

```
E-Kong/
├── app/
│   ├── models/
│   │   ├── stt.py          # Faster-Whisper STT（單例 + asyncio.Lock）
│   │   ├── llm.py          # GGUF LLM（llama.cpp，ThreadPoolExecutor）
│   │   ├── tts_zh.py       # ChatTTS 中文 TTS
│   │   └── tts_tw.py       # MMS-TTS 台語（facebook/mms-tts-nan）
│   ├── services/
│   │   ├── audio.py        # LINE 音訊下載 / ffmpeg 轉換 / TTS 上傳
│   │   ├── agent.py        # 情緒偵測 + ChatML Prompt + 對話歷史
│   │   └── tts.py          # TTS 路由 dispatcher
│   ├── config.py           # pydantic-settings 集中管理
│   ├── line_handler.py     # LINE 事件路由 + 簽名驗證
│   └── main.py             # FastAPI lifespan + /webhook + /health
├── scripts/
│   └── vram_profile.py     # VRAM 量測 + 調優建議
├── tests/
│   └── test_pipeline.py    # 30+ pytest 測試（全 Mock，不需 GPU）
├── setup_colab.py          # Colab 自動化啟動腳本
├── setup_ekong.ipynb       # Colab 一鍵啟動 notebook
├── requirements.txt
├── pytest.ini
└── .env.example
```

## 完整訊息流程

```
LINE 使用者
  │
  ▼ 傳送文字 or 語音
LINE Messaging API
  │
  ▼ Webhook POST /webhook
FastAPI (ngrok → Colab)
  │
  ├─[文字訊息]──▶ parse_command() → chat() → detect_emotion()
  │                                 → build_prompt() → LLM.generate()
  │                                 → try_reply_audio() → ChatTTS / MMS-TTS
  │
  └─[語音訊息]──▶ download_line_audio() → ffmpeg WAV
                  → Whisper.transcribe() → STT 文字
                  → chat() → LLM.generate()
                  → try_reply_audio() (語言路由: "nan"→MMS / 否則→ChatTTS)
                  → reply_audio_message() → LINE Audio Message
```

## 使用教學（快速開始）

### 1. 前置作業（取得 Tokens）
- **LINE Messaging API**：
  1. 至 [LINE Developers Console](https://developers.line.biz/) 建立 Provider 與 Channel (Messaging API)。
  2. 取得 **Channel Access Token**（需核發）與 **Channel Secret**（在 Basic settings）。
  3. 關閉「自動回覆訊息」與「問候訊息」功能（在官方帳號設定中）。
- **ngrok**：
  1. 註冊 [ngrok](https://ngrok.com/)。
  2. 至 Dashboard 取得 **Auth Token**。

### 2. 準備 GGUF 模型
1. 至 HuggingFace 下載 `taide-lx-7b-chat.Q4_K_M.gguf`（或 Llama 3 8B Instruct GGUF）。
2. 將模型上傳至你的 Google Drive 任意資料夾，例如：`我的雲端硬碟/ekong_models/`。

### 3. Colab 部署與啟動
1. 在瀏覽器開啟 Google Colab。
2. 匯入或上傳本專案的 `setup_ekong.ipynb`。
3. **設定 Colab Secrets**（左側側邊欄的鑰匙圖示 🔑）：
   - 新增 `LINE_CHANNEL_ACCESS_TOKEN` 並貼上你的 Token。
   - 新增 `LINE_CHANNEL_SECRET`。
   - 新增 `NGROK_AUTH_TOKEN`。
4. 設定 Runtime（執行階段）：選擇 **T4 GPU**。
5. **依序執行 Notebook 的 Cell 1 到 4**：
   - Cell 1：確認 GPU 狀態。
   - Cell 2：向 GitHub Clone 本專案。
   - Cell 3：自動生成 `.env` 設定檔（預設 LLM 路徑為 `/content/drive/MyDrive/ekong_models/taide-lx-7b-chat.Q4_K_M.gguf`，請依實際擺放位置修改 Cell 3 的路徑）。
   - Cell 4：執行環境建置（掛載 Google Drive、安裝依賴、啟動 FastAPI 與 ngrok 並自動註冊 LINE Webhook）。首次執行約需 5–8 分鐘。

### 4. 測試與使用
1. Notebook 啟動完成後會印出 🎉 服務已啟動，並顯示 Webhook URL。
2. 開啟你的 LINE，加入剛建立的機器人好友。
3. 傳送文字訊息或語音訊息，機器人就會用語音（中文 ChatTTS 或台語 MMS-TTS）並搭配文字回覆！

> **ℹ️ 開發者重啟技巧**：若中斷連線，只需重新執行 Notebook 的 **Cell 5 (快速重啟)** 即可，無須重新安裝套件。

## 環境變數（.env）

| 變數 | 說明 |
|------|------|
| `LINE_CHANNEL_ACCESS_TOKEN` | LINE Channel Access Token |
| `LINE_CHANNEL_SECRET` | LINE Channel Secret |
| `NGROK_AUTH_TOKEN` | ngrok 認證 Token |
| `LLM_MODEL_PATH` | GGUF 模型路徑（Google Drive） |
| `LLM_N_GPU_LAYERS` | GPU offload 層數（T4 建議 35） |
| `WHISPER_MODEL_SIZE` | `tiny` / `base` / `small` |

## VRAM 預算（T4 15GB）

| 模組 | 佔用 |
|------|------|
| Whisper base INT8 | ~300 MB |
| TAIDE-7B Q4_K_M (35 層) | ~4.8 GB |
| ChatTTS | ~1.5 GB |
| MMS-TTS | ~300 MB |
| 系統 overhead | ~1 GB |
| **合計** | **~8 GB ✅** |

## 測試執行

```bash
# 本機執行（不需 GPU / 模型檔）
pip install pytest pytest-asyncio
pytest tests/ -v

# Colab VRAM 量測
python scripts/vram_profile.py
python scripts/vram_profile.py --only stt llm  # 只測指定模組
```

## 健康檢查 API

`GET /health` 回傳：
```json
{
  "status": "ok",
  "service": "Ē-Kóng",
  "version": "0.1.0",
  "models": {
    "stt": "ok",
    "llm": "ok",
    "tts_zh": "ok",
    "tts_tw": "ok"
  },
  "gpu": {
    "device": "Tesla T4",
    "allocated_mb": "8192",
    "total_mb": "15360",
    "usage_pct": "53.3%"
  }
}
```

## 指令支援

| 使用者輸入 | 行為 |
|-----------|------|
| 任意文字 | 情緒偵測 → LLM → TTS 回覆 |
| 語音訊息 | STT → LLM → TTS 回覆（台語語音自動切 MMS-TTS）|
| `重設` / `reset` | 清除此使用者對話歷史 |

## Graceful Fallback 機制

- LLM 未載入 → Echo 模式（原文回傳）
- TTS 合成失敗 → 純文字回覆
- STT 未辨識到文字 → 提示重說
- 音訊下載失敗 → 友善錯誤訊息
