"""
app/services/agent.py — 排球錦標賽場控 Agent（重構版）
=====================================================
架構重點（與舊版重大差異）：
  1. 「光速 Regex 路由」取代第一輪 LLM：fast_intent_router()
     - 毫秒級意圖分類，不消耗 GPU / API 配額
  2. 單次 LLM 呼叫：build_final_prompt() 組裝完整 Context 後只打一次 Qwen
  3. 純文字輸出：不再有 TTS、不再有 AudioSendMessage
  4. Groq Whisper STT：語音辨識改由 Groq API 完成（見 line_handler.py）

Intent 分類（Regex 決定）：
  - Reject_Update      ：企圖修改比分 → 硬短路，不呼叫 LLM
  - Broadcast          ：廣播請求
  - Query_Score_Status ：即時比分查詢 → 呼叫 Google Sheets 工具
  - Query_Schedule     ：規則/場地/賽制查詢 → 讀取靜態 schedule.md
  - General_Chat       ：其餘 → 靜態手冊 + LLM 回答
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from loguru import logger

from app.models.llm import generate
from app.services.tools import (
    MatchQueryResult,
    format_match_result_for_llm,
    tool_query_elimination,
    tool_query_google_sheet,
    tool_query_groups,
    tool_query_loser_standings,
    tool_query_standings,
)


# ─── Intent 枚舉 ──────────────────────────────────────────────────────────────

class Intent(str, Enum):
    REJECT_UPDATE      = "Reject_Update"       # 企圖更新比分 → 攔截
    BROADCAST          = "Broadcast"           # 全場廣播
    QUERY_SCORE_STATUS = "Query_Score_Status"  # 即時比分查詢
    QUERY_SCHEDULE     = "Query_Schedule"      # 靜態賽事規則查詢
    GENERAL_CHAT       = "General_Chat"        # 一般詢問


# ─── Agent 回應結構 ────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    intent:        Intent
    response_text: str
    action:        dict | None  # 僅 Broadcast 有值，其餘 None


# ─── 靜態賽事手冊載入 ──────────────────────────────────────────────────────────

_SCHEDULE_MD_PATH = Path(__file__).parent.parent / "data" / "schedule.md"


def _load_schedule_md() -> str:
    """啟動時載入 schedule.md，失敗時回傳空字串（不阻斷服務）。"""
    try:
        text = _SCHEDULE_MD_PATH.read_text(encoding="utf-8")
        logger.info(f"✅ schedule.md 載入成功（{len(text)} 字元）")
        return text
    except FileNotFoundError:
        logger.warning(f"⚠️  找不到 {_SCHEDULE_MD_PATH}，靜態賽事資訊將缺失。")
        return ""


_SCHEDULE_STATIC: str = _load_schedule_md()

# ─── 拒絕更新文字（硬短路，不過 LLM）────────────────────────────────────────

_REJECT_UPDATE_TEXT = (
    "抱歉，語音助理目前僅供查詢，"
    "若需修改比分請洽紀錄台工作人員。"
)


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 光速 Regex 路由
# ═══════════════════════════════════════════════════════════════════════════════

# 各 Intent 的 Regex pattern（順序決定優先權：Reject 最高）
_PATTERNS: list[tuple[Intent, re.Pattern[str]]] = [
    (
        Intent.REJECT_UPDATE,
        re.compile(r"(改|修改|更新|刪除|填入|記錄|紀錄|寫入).{0,8}(比分|分數|賽程|場次|局數)", re.IGNORECASE),
    ),
    (
        Intent.BROADCAST,
        re.compile(r"廣播", re.IGNORECASE),
    ),
    (
        Intent.QUERY_SCORE_STATUS,
        re.compile(
            r"(比分|幾比幾|幾局|幾分|比幾|分數|現在.*?(贏|輸|領先)|誰贏|誰輸|賽程|場次|結果|狀態|比賽)",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.QUERY_SCHEDULE,
        re.compile(
            r"(規則|網高|資格|報名|停車|場地|積分怎麼算|賽制|時間|幾點|幾號|分組|哪些隊|參賽隊|晉級|淘汰|幾隊)",
            re.IGNORECASE,
        ),
    ),
]


def fast_intent_router(user_text: str) -> dict:
    """
    光速 Regex 意圖路由器。

    Parameters
    ----------
    user_text : str
        使用者輸入（STT 轉錄或直接文字）。

    Returns
    -------
    dict
        {"intent": Intent, "reject_msg": str | None}
        reject_msg 僅在 Reject_Update 時有值，可直接回傳給使用者。
    """
    for intent, pattern in _PATTERNS:
        if pattern.search(user_text):
            logger.debug(f"🎯 Regex 路由命中 → {intent.value}（input: {user_text!r:.60}）")
            if intent == Intent.REJECT_UPDATE:
                return {"intent": intent, "reject_msg": _REJECT_UPDATE_TEXT}
            return {"intent": intent, "reject_msg": None}

    # 沒有任何 pattern 命中 → 一般詢問
    logger.debug(f"🎯 Regex 路由 fallback → General_Chat（input: {user_text!r:.60}）")
    return {"intent": Intent.GENERAL_CHAT, "reject_msg": None}


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 Prompt 組裝
# ═══════════════════════════════════════════════════════════════════════════════

_BASE_SYSTEM_PROMPT = """\
你是「賽場助理」，協助主辦人管控大型排球錦標賽現場（約 500 人規模）。
你的輸入可能是語音辨識（STT）的轉錄結果，可能有錯字，請自行判斷語意。

【回答規則——嚴格執行】
1. 只能依據下方提供的【賽事手冊】與【即時查詢資料】回答，不可自行捏造資料。
2. 如果找不到答案，請說：「這個問題我不清楚，建議直接問報到台工作人員喔！」
3. 回覆必須 50 字以內，口語化，像在現場說話一樣，不要用表格或 Markdown 格式。
4. 不要說你是 AI 或語言模型。

"""


def build_final_prompt(
    user_text: str,
    intent: Intent,
    dynamic_data: str = "",
) -> str:
    """
    組裝最終送給 LLM 的完整 Prompt（ChatML 格式）。

    Parameters
    ----------
    user_text : str
        使用者原始輸入。
    intent : Intent
        Regex 路由決定的意圖（影響 System Prompt 的說明文字）。
    dynamic_data : str
        即時查詢資料（Query_Score_Status 時為 Google Sheets 結果；其餘為空）。

    Returns
    -------
    str
        完整的 ChatML prompt 字串。
    """
    # 組裝 System Prompt
    system = _BASE_SYSTEM_PROMPT

    # 永遠注入靜態賽事手冊
    if _SCHEDULE_STATIC:
        system += f"【賽事手冊（規則、場地、賽制、報名等靜態資訊）】\n{_SCHEDULE_STATIC}\n\n"

    # 若有即時查詢資料，額外注入
    if dynamic_data:
        system += f"【即時查詢資料（優先依此回答比分相關問題）】\n{dynamic_data}\n\n"

    # 根據 intent 給 LLM 額外提示
    if intent == Intent.BROADCAST:
        system += "【任務】使用者想廣播，請協助將廣播內容整理為清楚的廣播文稿（繁體中文，口語化）。\n"
    elif intent == Intent.QUERY_SCORE_STATUS:
        system += "【任務】使用者詢問比分或賽況，請依據即時查詢資料回答。若查無資料請誠實告知。\n"
    elif intent == Intent.QUERY_SCHEDULE:
        system += "【任務】使用者詢問賽事規則或靜態資訊，請依據賽事手冊回答。\n"
    else:  # General_Chat
        system += "【任務】使用者有一般詢問，請依據賽事手冊回答；若手冊無資料則說不清楚。\n"

    return (
        f"<|im_start|>system\n{system.strip()}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 工具呼叫
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_score_context(user_text: str) -> str:
    """
    呼叫 Google Sheets 工具，以使用者原文作為 match_id hint。

    注意：tool_query_google_sheet 內部會做模糊比對，
    傳入完整使用者文字即可，讓工具自行篩選有效欄位。
    """
    result: MatchQueryResult = tool_query_google_sheet(user_text)
    context = format_match_result_for_llm(result)
    logger.info(f"🔍 Google Sheets 查詢結果｜{context[:100]}")
    return context


# ═══════════════════════════════════════════════════════════════════════════════
# 🤖 Agent 主流程（單輪 LLM）
# ═══════════════════════════════════════════════════════════════════════════════

async def chat(user_id: str, user_input: str) -> AgentResponse:
    """
    完整 Agent 推論流程（單輪 LLM）：

    1. fast_intent_router      → Regex 意圖分類（毫秒級）
    2. Reject_Update           → 硬短路，不呼叫 LLM
    3. Query_Score_Status      → 呼叫 Google Sheets 工具取得 dynamic_data
    4. build_final_prompt      → 組裝完整 Prompt（含 schedule.md + dynamic_data）
    5. generate()              → 單次 LLM 推論
    6. 回傳 AgentResponse（純文字）
    """
    logger.info(f"💬 Agent.chat｜user={user_id}｜input={user_input!r:.80}")

    # ── Step 1: Regex 路由 ──────────────────────────────────────────────────
    route = fast_intent_router(user_input)
    intent: Intent = route["intent"]

    # ── Step 2: Reject_Update 硬短路 ───────────────────────────────────────
    if intent == Intent.REJECT_UPDATE:
        logger.warning(f"🚫 Reject_Update 攔截｜{user_id}｜{user_input!r:.80}")
        return AgentResponse(
            intent=Intent.REJECT_UPDATE,
            response_text=_REJECT_UPDATE_TEXT,
            action=None,
        )

    # ── Step 3: 工具呼叫（視 intent 決定）──────────────────────────────────
    dynamic_data = ""
    if intent == Intent.QUERY_SCORE_STATUS:
        dynamic_data = _fetch_score_context(user_input)

    # ── Step 4: 組裝 Prompt ──────────────────────────────────────────────
    prompt = build_final_prompt(user_input, intent, dynamic_data)
    logger.debug(f"📝 Prompt 長度：{len(prompt)} chars")

    # ── Step 5: 單次 LLM 推論 ────────────────────────────────────────────
    try:
        raw = await generate(prompt)
        response_text = raw.strip()
        # 防止空字串導致 LINE API 400
        if not response_text:
            response_text = "抱歉，我剛才沒理解，可以再說一次嗎？"
    except Exception as exc:
        logger.error(f"❌ LLM 推論失敗：{exc}")
        raise RuntimeError("LLM 推論失敗") from exc

    logger.info(f"✅ Agent 回覆｜intent={intent.value}｜{response_text[:80]}")

    return AgentResponse(
        intent=intent,
        response_text=response_text,
        action=None,  # Broadcast 的 action 目前由 line_handler 處理廣播文字即可
    )


# ─── 指令解析（維持原有 reset 功能）──────────────────────────────────────────

_RESET_KEYWORDS = {"重設", "重置", "清除記憶", "忘掉之前", "forget", "reset"}


def parse_command(text: str) -> str | None:
    normalized = text.strip().lower()
    for kw in _RESET_KEYWORDS:
        if kw in normalized:
            return "reset"
    return None
