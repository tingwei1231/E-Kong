# Project Ē-Kóng (會講) v2 — 開發 Walkthrough

> **架構版本**：v2（語音輸入 × 文字輸出）
> **重構日期**：2026-04-10
> **主要變更**：本地 Whisper STT → Groq API；移除 TTS；引入光速 Regex 路由

---

## 完整目錄結構

```
E-Kong/
├── app/
│   ├── models/
│   │   └── llm.py          # GGUF LLM 單例（llama.cpp，ThreadPoolExecutor）
│   ├── services/
│   │   ├── audio.py        # LINE 音訊下載 / ffmpeg 轉換（m4a → WAV）
│   │   ├── agent.py        # Regex 路由 + build_final_prompt + 單輪 LLM
│   │   └── tools.py        # Google Sheets CSV 查詢工具群
│   ├── data/
│   │   └── schedule.md     # 靜態賽事手冊（規則/場地/賽制）
│   ├── config.py           # pydantic-settings 集中管理
│   ├── line_handler.py     # LINE 事件路由 + Groq STT + 簽名驗證
│   └── main.py             # FastAPI lifespan + /webhook + /health
├── tests/
│   └── test_pipeline.py    # pytest 測試（Mock 版，不需 GPU）
├── setup_colab.py          # Colab 自動化啟動腳本（v2）
├── setup_ekong.ipynb       # Colab 一鍵啟動 Notebook
├── requirements.txt
├── pytest.ini
├── .env.example
└── walkthrough.md
```

---

## 訊息流程（v2）

```
LINE 使用者
  │
  ▼ 傳送文字 or 語音
LINE Messaging API
  │
  ▼ Webhook POST /webhook
FastAPI (ngrok → Colab) — 立即回 200 OK
  │
  ├─[文字訊息]─▶ fast_intent_router() [Regex，毫秒級]
  │               │
  │               ├─ Reject_Update ──▶ 直接拒絕文字（不過 LLM）
  │               ├─ Broadcast     ──▶ build_final_prompt() → LLM → push_text
  │               ├─ Query_Score   ──▶ tool_query_google_sheet() → LLM → push_text
  │               ├─ Query_Schedule──▶ schedule.md → LLM → push_text
  │               └─ General_Chat  ──▶ schedule.md → LLM → push_text
  │
  └─[語音訊息]─▶ download_line_audio() → ffmpeg WAV
                  │
                  ▼ asyncio.to_thread（不阻塞 event loop）
                 Groq API whisper-large-v3 (STT)
                  │
                  ▼ 文字結果
                 fast_intent_router() → 同上文字流程
```

---

## 快速開始

### 1. 取得所需 Token / Key

| 服務 | 取得方式 |
|------|---------|
| LINE Channel Access Token | [LINE Developers Console](https://developers.line.biz/) → Messaging API Channel |
| LINE Channel Secret | 同上 → Basic settings |
| ngrok Auth Token | [ngrok Dashboard](https://dashboard.ngrok.com/get-started/your-authtoken) |
| **Groq API Key** | [console.groq.com](https://console.groq.com/) → 免費方案：2000 分鐘/天 |

> **重要**：關閉 LINE 官方帳號的「自動回覆」與「問候訊息」功能。

### 2. 準備 LLM 模型（GGUF）

1. 從 HuggingFace 下載量化模型，例如：
   - `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`（輕量，CPU 即可）
   - `Qwen2.5-7B-Instruct-Q4_K_M.gguf`（效果更佳，T4 GPU 推薦）
2. 上傳至 Google Drive：`我的雲端硬碟/ekong_models/`

### 3. Colab 部署

1. 開啟 [Google Colab](https://colab.google/)，Runtime 選 **T4 GPU**。
2. Clone 本專案：
   ```bash
   !git clone https://github.com/YOUR_USERNAME/E-Kong.git
   %cd E-Kong
   ```
3. 設定 **Colab Secrets**（左側 🔑 圖示）：
   ```
   LINE_CHANNEL_ACCESS_TOKEN  = <你的 Token>
   LINE_CHANNEL_SECRET        = <你的 Secret>
   NGROK_AUTH_TOKEN           = <你的 ngrok Token>
   GROQ_API_KEY               = <你的 Groq Key>   ← v2 新增，必填
   ```
4. 建立 `.env`：
   ```python
   import os, json
   from google.colab import userdata

   env = {
       "LINE_CHANNEL_ACCESS_TOKEN": userdata.get("LINE_CHANNEL_ACCESS_TOKEN"),
       "LINE_CHANNEL_SECRET":       userdata.get("LINE_CHANNEL_SECRET"),
       "NGROK_AUTH_TOKEN":          userdata.get("NGROK_AUTH_TOKEN"),
       "GROQ_API_KEY":              userdata.get("GROQ_API_KEY"),
       "LLM_MODEL_PATH":            "/content/drive/MyDrive/ekong_models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
       "LLM_N_GPU_LAYERS":          "35",
       "LLM_MAX_TOKENS":            "512",
       "LLM_TEMPERATURE":           "0.7",
       # Google Sheets CSV 網址（依實際填入）
       "GOOGLE_SHEET_CSV_SCORE":    "",
       "GOOGLE_SHEET_CSV_GROUPS":   "",
       "GOOGLE_SHEET_CSV_STANDINGS":"",
       "GOOGLE_SHEET_CSV_LOSER_STANDINGS": "",
       "GOOGLE_SHEET_CSV_ELIMINATION":     "",
   }
   with open(".env", "w") as f:
       for k, v in env.items():
           f.write(f"{k}={v}\n")
   print(".env 建立完成")
   ```
5. 啟動系統：
   ```bash
   !python setup_colab.py
   ```

### 4. 測試指令範例

| 輸入類型 | 範例 | 預期行為 |
|---------|------|---------|
| 文字 | `「A組第一場現在比分幾比幾？」` | 查詢 Google Sheets → 文字推播比分 |
| 文字 | `「報名截止時間是幾點？」` | 讀 schedule.md → 文字回答 |
| 文字 | `「廣播：請108號選手到報到台」` | LLM 整理廣播稿 → 文字回覆 |
| 文字 | `「幫我把第一場改成台大勝」` | Regex 直接攔截 → 拒絕，不過 LLM |
| 語音 | 說「A組第一場比分多少」 | Groq STT → 同上文字流程 |
| 文字 | `reset` | 清除對話記憶 |

---

## 環境變數（.env）

| 變數 | 必填 | 說明 |
|------|:----:|------|
| `LINE_CHANNEL_ACCESS_TOKEN` | ✅ | LINE Channel Access Token |
| `LINE_CHANNEL_SECRET` | ✅ | LINE Channel Secret |
| `NGROK_AUTH_TOKEN` | ✅ | ngrok 認證 Token |
| `GROQ_API_KEY` | ✅ | Groq API Key（STT 使用） |
| `LLM_MODEL_PATH` | ✅ | GGUF 模型路徑 |
| `LLM_N_GPU_LAYERS` | | GPU offload 層數（T4 建議 35，CPU 改 0） |
| `GOOGLE_SHEET_CSV_SCORE` | | 賽程比分 CSV 公開網址 |
| `GOOGLE_SHEET_CSV_GROUPS` | | 分組名單 CSV 公開網址 |
| `GOOGLE_SHEET_CSV_STANDINGS` | | 積分表 CSV 公開網址 |
| `GOOGLE_SHEET_CSV_LOSER_STANDINGS` | | 敗者組積分 CSV 公開網址 |
| `GOOGLE_SHEET_CSV_ELIMINATION` | | 晉淘結果 CSV 公開網址 |

---

## VRAM 預算（v2 vs v1）

| 模組 | v1 佔用 | v2 佔用 |
|------|:-------:|:-------:|
| Whisper STT | ~300 MB | **0 MB**（Groq API）|
| Qwen 1.5B Q4 (35 層) | – | ~2.5 GB |
| ChatTTS | ~1.5 GB | **0 MB**（已移除）|
| MMS-TTS 台語 | ~300 MB | **0 MB**（已移除）|
| LLM（TAIDE 7B Q4）| ~4.8 GB | 可選 |
| **合計（1.5B 版）** | **~8 GB** | **~2.5 GB ✅** |

> v2 顯著降低 VRAM 需求，**CPU-only Colab 也能跑**（使用 CPU 版 llama-cpp + Groq API）

---

## 意圖路由邏輯（Regex 優先權）

```
輸入文字
  │
  ├─ 1. Reject_Update  ← (改|修改|更新|刪除|填入|記錄).*(比分|賽程)
  ├─ 2. Broadcast      ← 廣播
  ├─ 3. Query_Score    ← (比分|幾比幾|誰贏|賽程|比賽|結果|狀態…)
  ├─ 4. Query_Schedule ← (規則|網高|資格|報名|停車|場地|積分怎麼算…)
  └─ 5. General_Chat   ← 無命中（fallback）
```

**Reject_Update 優先权最高**：確保「修改比分廣播給大家」不會誤觸 Broadcast。

---

## 健康檢查 API

```
GET /health
```

回傳範例（v2）：
```json
{
  "status": "ok",
  "service": "Ē-Kóng",
  "version": "2.0.0",
  "models": {
    "llm": "ok",
    "stt_groq": "key_set",
    "tts": "disabled (text-only mode)"
  },
  "gpu": {
    "device": "Tesla T4",
    "allocated_mb": "2560",
    "total_mb": "15360",
    "usage_pct": "16.7%"
  }
}
```

---

## 測試執行

```bash
# 本機執行（不需 GPU / 模型檔）
pip install pytest pytest-asyncio
pytest tests/ -v

# 快速驗證 Regex 路由（不需任何服務）
python -c "
from app.services.agent import fast_intent_router
tests = [
    '幫我把第一場改成台大勝',     # → Reject_Update
    '廣播：請108號到報到台',       # → Broadcast
    'A組第一場現在比分多少',       # → Query_Score_Status
    '報名截止時間是幾點',          # → Query_Schedule
    '廁所在哪裡',                  # → General_Chat
]
for t in tests:
    r = fast_intent_router(t)
    print(f'{r[\"intent\"].value:25} ← {t}')
"
```

---

## Graceful Fallback 機制

| 情境 | 處理方式 |
|------|---------|
| LLM 未載入 | Echo 模式（原文推播） |
| Groq STT 失敗 | 錯誤訊息推播 + 建議改文字輸入 |
| 語音辨識為空（靜音） | 提示「訊號不清楚，請再說一次」 |
| Reject_Update | 硬短路，不過 LLM，直接推播拒絕訊息 |
| Google Sheets 查詢失敗 | LLM 誠實告知無資料（schedule.md 仍可用） |
| 音訊下載失敗 | 友善錯誤訊息 |
