"""
app/services/agent.py — 排球錦標賽語音場控 Agent
=================================================
職責：
  1. 載入 app/data/schedule.md（靜態賽事資訊，注入 System Prompt）
  2. 對話歷史管理（In-memory，per-user，環形緩衝）
  3. 系統 Prompt 組裝（靜態手冊 + JSON 格式指示）
  4. 呼叫 llm.generate() 取得 JSON 回覆，並解析路由
  5. 意圖防護：Reject_Update 攔截所有語音更新請求

五種 Intent：
  - Query_Score_Status：查詢即時比分（→ tool_query_google_sheet）
  - Query_Schedule    ：查詢賽程/分組/積分/晉淘（→ tool_query_schedule 等）
  - Broadcast         ：全場廣播
  - Reject_Update     ：企圖更新比分 → 硬短路
  - General_Chat      ：一般詢問（直接由 LLM + schedule.md 回答）

Context 架構：
  ┌─ System Prompt（固定）
  │    ├─ 角色定義 + 五種 Intent 說明
  │    └─ schedule.md 靜態全文（啟動時一次載入）
  │
  └─ Tool Context（動態，每輪請求按需查詢）
       ├─ Query_Score_Status → 即時比分
       └─ Query_Schedule     → 賽程/分組/積分/晉淘
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Deque

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
    QUERY_SCORE_STATUS = "Query_Score_Status"  # 即時比分查詢（唯讀）
    QUERY_SCHEDULE     = "Query_Schedule"       # 賽程/分組/積分/晉淘查詢
    BROADCAST          = "Broadcast"            # 全場廣播
    REJECT_UPDATE      = "Reject_Update"        # 企圖更新比分 → 攔截
    GENERAL_CHAT       = "General_Chat"         # 一般詢問（用 schedule.md 回答）


# ─── Agent 回應結構 ────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    intent:        Intent
    response_text: str
    action:        dict | None  # 僅 Broadcast 有值，其餘 None


# ─── 對話歷史 ─────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str
    content: str


@dataclass
class ConversationHistory:
    maxlen: int = 10
    turns: Deque[Turn] = field(default_factory=lambda: deque(maxlen=10))

    def add_user(self, text: str) -> None:
        self.turns.append(Turn(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self.turns.append(Turn(role="assistant", content=text))

    def clear(self) -> None:
        self.turns.clear()


_histories: dict[str, ConversationHistory] = {}


def get_history(user_id: str) -> ConversationHistory:
    if user_id not in _histories:
        _histories[user_id] = ConversationHistory()
    return _histories[user_id]


def clear_history(user_id: str) -> None:
    if user_id in _histories:
        _histories[user_id].clear()
        logger.info(f"🗑️  已清除 {user_id} 的對話歷史。")


# ─── 靜態賽事手冊載入 ────────────────────────────────────────────────────────

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


# ─── System Prompt ────────────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = """\
你是「賽場助理」，一個專業、俐落的大型排球錦標賽現場語音場控 Agent。
你服務的對象是賽事主辦人，協助他透過語音指令管控近 500 人規模的賽事現場。
你收到的輸入是語音辨識（STT）的轉錄結果，可能有錯字或語音切割不完整，請自行判斷語意。

【意圖分類 (intent)】嚴格只能從以下五種選一：

- "Query_Score_Status"：即時比分查詢
  使用者想查詢某場次的當前比分、勝負（e.g.「A組第一場現在幾比幾？」「誰贏了？」）
  → 後端工具會自動查詢 Google Sheets 的「賽程」分頁。

- "Query_Schedule"：賽程資訊查詢
  使用者詢問分組名單、正規積分排名、敗者組積分、晉級或淘汰結果（e.g.「B組有哪些隊？」
  「目前積分第一是誰？」「誰晉級了？」「敗者組積分怎麼算？」）
  → 後端工具會依查詢種類自動查詢對應的 Sheets 分頁。

- "Broadcast"：全場廣播
  需要廣播給全場（e.g.「通知大家下午兩點集合」「尋找報名108號選手」）

- "Reject_Update"：【最高優先權，嚴格執行】
  使用者企圖透過語音「更新、修改、填入、刪除、紀錄」任何比分或賽程資料時，
  一律分類為此意圖（e.g.「台大對政大剛剛25比18」「幫我把第一場改成台大勝」）。

- "General_Chat"：一般詢問或閒聊
  其他問題（e.g.「廁所在哪？」「什麼時候頒獎？」）→ 優先使用下方靜態賽事手冊回答。
  
【安全規則】
- 任何涉及「寫入」「更新」「修改」「紀錄」「填入」比分或賽程的指令，一律為 Reject_Update。
- 語音助理嚴格唯讀，不得協助任何資料寫入。

【輸出格式規定】必須是合法 JSON，包含以下欄位：
1. "intent"：以上五種之一。
2. "response_text"：給主辦人聽的語音回覆，100 字以內，專業簡潔。
   - Reject_Update：「抱歉，為確保賽事公平性，語音助理目前僅開放比分查詢，若需修改請洽紀錄台工作人員。」
   - Query_Score_Status/Query_Schedule：根據 [工具查詢結果] 撰寫自然語句。
3. "action"：
   - Broadcast：{"target": "all" | "<編號>", "message": "<廣播全文>"}
   - 其餘一律 null
4. "query_params"（僅 Query_Score_Status / Query_Schedule 需要）：
   - Query_Score_Status：{"match_id": "<場次ID>"}
   - Query_Schedule：{"type": "groups"|"standings"|"loser_standings"|"elimination", "group": "<組別名稱或null>"}

【絕對禁止】
- 不可在 JSON 之外輸出任何文字、解釋或 Markdown
- 不可揭露你是 AI 或語言模型
- action、query_params 欄位不可缺少（無值時設為 null）

"""

_REJECT_UPDATE_TEXT = (
    "抱歉，為確保賽事公平性，語音助理目前僅開放比分查詢，"
    "若需修改請洽紀錄台工作人員。"
)


def _build_system_prompt() -> str:
    """組裝含靜態賽事手冊的完整 System Prompt。"""
    static_section = (
        f"\n【靜態賽事手冊（請優先依此回答賽制、地點、規則等問題）】\n{_SCHEDULE_STATIC}\n"
        if _SCHEDULE_STATIC else ""
    )
    return _BASE_SYSTEM_PROMPT.strip() + static_section


_FULL_SYSTEM_PROMPT: str = _build_system_prompt()


def build_prompt(
    user_id: str,
    user_input: str,
    tool_context: str | None = None,
) -> str:
    """組裝完整 ChatML 格式 Prompt，動態注入工具查詢結果。"""
    history = get_history(user_id)
    system  = _FULL_SYSTEM_PROMPT
    if tool_context:
        system = system.rstrip() + f"\n\n【本輪工具查詢結果（優先依此回答）】\n{tool_context}\n"

    parts: list[str] = [f"<|im_start|>system\n{system}<|im_end|>\n"]
    for turn in history.turns:
        parts.append(f"<|im_start|>{turn.role}\n{turn.content}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user_input}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# ─── JSON 解析 ────────────────────────────────────────────────────────────────

_JSON_EXTRACT_RE  = re.compile(r"\{.*?\}", re.DOTALL)
_MD_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

_FALLBACK_RESPONSE = AgentResponse(
    intent=Intent.GENERAL_CHAT,
    response_text="抱歉，我剛才沒聽清楚，請再說一次。",
    action=None,
)


def parse_llm_json(raw: str) -> tuple[AgentResponse, dict]:
    """
    解析 LLM 輸出 JSON，回傳 (AgentResponse, query_params)。
    容錯：解析失敗時回傳 fallback General_Chat。
    """
    if not raw.strip():
        return _FALLBACK_RESPONSE, {}

    md_m = _MD_CODE_BLOCK_RE.search(raw)
    json_str = md_m.group(1) if md_m else (
        _JSON_EXTRACT_RE.search(raw).group(0)
        if _JSON_EXTRACT_RE.search(raw) else raw
    )

    try:
        data = json.loads(json_str)
        intent_val = str(data.get("intent", "General_Chat")).strip()
        matched_intent = None
        for member in Intent:
            if member.value.lower() == intent_val.lower():
                matched_intent = member
                break
        intent = matched_intent if matched_intent else Intent.GENERAL_CHAT

        query_params = data.get("query_params") or {}
        response_text = (data.get("response_text") or "").strip()
        
        # 防止空字串導致 LINE API 回傳 400 錯誤
        if not response_text:
            response_text = "好的，馬上為您查詢。" if intent != Intent.GENERAL_CHAT else _FALLBACK_RESPONSE.response_text

        return AgentResponse(
            intent=intent,
            response_text=response_text,
            action=data.get("action"),
        ), query_params

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning(f"⚠️  JSON 解析失敗（{exc}）｜raw={raw!r:.120}")
        return _FALLBACK_RESPONSE, {}


# ─── 動態工具路由 ─────────────────────────────────────────────────────────────

def _run_schedule_tool(query_params: dict) -> str:
    """
    根據 LLM 輸出的 query_params 呼叫對應的賽程查詢工具。

    query_params 格式：
      {"type": "schedule"|"groups"|"standings"|"elimination",
       "day":   "<第1天/第2天>" | null,
       "group": "<A組/B組/…>"  | null}
    """
    qtype = (query_params.get("type") or "schedule").lower()
    day   = query_params.get("day")   or None
    group = query_params.get("group") or None

    if qtype == "groups":
        return tool_query_groups(group)
    elif qtype == "standings":
        return tool_query_standings(group)
    elif qtype == "elimination":
        return tool_query_elimination(group)
    else:  # default: schedule
        return tool_query_schedule(day)


# ─── Agent 主流程 ─────────────────────────────────────────────────────────────

async def chat(user_id: str, user_input: str) -> AgentResponse:
    """
    完整 Agent 推論流程（兩輪 LLM）：

    第一輪 → intent 分類 + query_params 萃取
        ├─ Reject_Update → 硬短路
        ├─ Query_Score_Status → tool_query_google_sheet(match_id)
        │                     → 第二輪 LLM（注入比分結果）
        ├─ Query_Schedule     → _run_schedule_tool(query_params)
        │                     → 第二輪 LLM（注入查詢結果）
        └─ Broadcast / General_Chat → 直接輸出（General_Chat 已有 schedule.md）
    """
    logger.debug(f"💬 Agent｜{user_id}｜歷史輪數={len(get_history(user_id).turns)}")

    # ── 第一輪 LLM：intent 分類 ──────────────────────────────────────────────
    prompt_classify     = build_prompt(user_id, user_input)
    raw_classify        = await generate(prompt_classify)
    agent_resp, qparams = parse_llm_json(raw_classify)

    logger.info(
        f"🎯 intent={agent_resp.intent.value}｜"
        f"qparams={qparams}｜action={agent_resp.action}｜"
        f"{agent_resp.response_text[:60]}"
    )

    # ── Reject_Update 防護（硬短路）────────────────────────────────────────
    if agent_resp.intent == Intent.REJECT_UPDATE:
        logger.warning(f"🚫 Reject_Update 攔截｜{user_id}｜{user_input!r:.80}")
        return AgentResponse(
            intent=Intent.REJECT_UPDATE,
            response_text=_REJECT_UPDATE_TEXT,
            action=None,
        )

    # ── 工具呼叫 ──────────────────────────────────────────────────────────
    tool_context: str | None = None

    if agent_resp.intent == Intent.QUERY_SCORE_STATUS:
        match_id = qparams.get("match_id") or user_input
        result: MatchQueryResult = tool_query_google_sheet(match_id)
        tool_context = format_match_result_for_llm(result)
        logger.info(f"🔍 比分查詢｜match_id={match_id}｜{tool_context[:80]}")

    elif agent_resp.intent == Intent.QUERY_SCHEDULE:
        tool_context = _run_schedule_tool(qparams)
        logger.info(f"📅 賽程查詢｜type={qparams.get('type')}｜{tool_context[:80]}")

    # ── 第二輪 LLM（僅工具查詢後才需要）──────────────────────────────────
    if tool_context:
        prompt_final       = build_prompt(user_id, user_input, tool_context=tool_context)
        raw_final          = await generate(prompt_final)
        agent_resp, _      = parse_llm_json(raw_final)
        logger.info(f"🗣️  最終回覆｜{agent_resp.response_text[:80]}")

    # ── 更新對話歷史 ────────────────────────────────────────────────────────
    history = get_history(user_id)
    history.add_user(user_input)
    history.add_assistant(agent_resp.response_text)

    return agent_resp


# ─── 指令解析 ─────────────────────────────────────────────────────────────────

_RESET_KEYWORDS = {"重設", "重置", "清除記憶", "忘掉之前", "forget", "reset"}


def parse_command(text: str) -> str | None:
    normalized = text.strip().lower()
    for kw in _RESET_KEYWORDS:
        if kw in normalized:
            return "reset"
    return None
