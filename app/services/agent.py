"""
app/services/agent.py — 情緒感知 LLM Agent 服務
=================================================
職責：
  1. 對話歷史管理（In-memory，per-user，環形緩衝）
  2. 情緒偵測（關鍵詞 + 標點啟發式規則，輕量快速）
  3. 系統 Prompt 組裝（含角色設定 + JSON 輸出格式要求）
  4. 呼叫 llm.generate() 取得 JSON 回覆，並解析路由

設計原則：
  - 歷史記錄以 deque(maxlen=N) 實作環形緩衝，防記憶體無限增長
  - LLM 強制輸出 JSON {intent, user_emotion, response_language, response_text}
  - intent="translate" → 直接翻譯；intent="companion" → 情緒陪伴回覆
  - TAIDE / Llama-3 均使用 ChatML 格式（<|im_start|> … <|im_end|>）

Prompt 格式（ChatML）：
  <|im_start|>system
  {system_prompt}<|im_end|>
  <|im_start|>user
  {user_msg}<|im_end|>
  <|im_start|>assistant
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque

from loguru import logger

from app.models.llm import generate


# ─── 情緒枚舉 ──────────────────────────────────────────────────────────────────

class Emotion(str, Enum):
    """偵測到的情緒類別。"""
    HAPPY     = "happy"
    SAD       = "sad"
    ANXIOUS   = "anxious"
    ANGRY     = "angry"
    LONELY    = "lonely"
    EXHAUSTED = "exhausted"
    NEUTRAL   = "neutral"

    @property
    def zh(self) -> str:
        """中文標籤（用於 Prompt 組裝）。"""
        return {
            "happy":     "開心",
            "sad":       "難過",
            "anxious":   "焦慮",
            "angry":     "生氣",
            "lonely":    "孤獨",
            "exhausted": "疲憊",
            "neutral":   "平靜",
        }[self.value]

    @property
    def emoji(self) -> str:
        return {
            "happy": "😊", "sad": "😢", "anxious": "😰",
            "angry": "😤", "lonely": "🥺", "exhausted": "😩",
            "neutral": "😌",
        }[self.value]


# ─── Intent 枚舉 ──────────────────────────────────────────────────────────────

class Intent(str, Enum):
    """使用者意圖類別。"""
    TRANSLATE = "translate"   # 明確要求翻譯
    COMPANION = "companion"   # 閒聊 / 情緒陪伴


# ─── Agent 回應結構 ────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    """解析後的 LLM JSON 輸出。"""
    intent:            Intent
    user_emotion:      str      # LLM 自行判斷的情緒字串（如 "sad", "exhausted"）
    response_language: str      # "zh-TW" | "nan"
    response_text:     str      # 實際回覆內容
    emotion:           Emotion  # 本地 heuristic 偵測結果（保留供 TTS 參考）


# ─── 情緒偵測（輕量 heuristic，Step 4 後可替換 BERT 分類器）───────────────────

# 關鍵詞對應情緒類別（中文 + 台語羅馬拼音常見詞）
_EMOTION_KEYWORDS: dict[Emotion, list[str]] = {
    Emotion.HAPPY:     ["開心", "高興", "快樂", "棒", "太好了", "哈哈", "😄", "🎉", "爽", "讚"],
    Emotion.SAD:       ["難過", "傷心", "哭", "可憐", "失落", "沮喪", "心痛", "痛苦", "😢", "😭"],
    Emotion.ANXIOUS:   ["擔心", "焦慮", "緊張", "不安", "怕", "害怕", "恐慌", "壓力", "😰", "😨"],
    Emotion.ANGRY:     ["氣", "生氣", "憤怒", "煩", "討厭", "幹", "靠", "怒", "😤", "😠"],
    Emotion.LONELY:    ["寂寞", "孤獨", "孤單", "沒人", "沒朋友", "想你", "思念", "🥺", "😔"],
    Emotion.EXHAUSTED: ["累", "好累", "疲憊", "沒力", "撐不住", "精疲力竭", "😩", "🥱"],
}

# 問句 / 感嘆號多 → 焦慮傾向
_ANXIOUS_PATTERN = re.compile(r"[？?!！]{2,}")


def detect_emotion(text: str) -> Emotion:
    """
    對輸入文字做輕量情緒偵測，回傳最可能的情緒類別。

    演算法：關鍵詞命中計分，最高分勝；平手或零命中回傳 NEUTRAL。
    可在 Step 4 替換為 BERT 情緒分類模型以提升準確度。

    Parameters
    ----------
    text : str
        使用者輸入文字（STT 轉錄或直接文字訊息）。

    Returns
    -------
    Emotion
        情緒類別。
    """
    scores: dict[Emotion, int] = {e: 0 for e in Emotion}

    for emotion, keywords in _EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[emotion] += 1

    # 焦慮規則：多個問號/感嘆號
    if _ANXIOUS_PATTERN.search(text):
        scores[Emotion.ANXIOUS] += 1

    best_emotion = max(scores, key=lambda e: scores[e])
    if scores[best_emotion] == 0:
        return Emotion.NEUTRAL

    logger.debug(f"🎭 情緒偵測：{best_emotion.zh}（{best_emotion.emoji}）｜scores={dict(scores)}")
    return best_emotion


# ─── 對話歷史 ─────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """單輪對話。"""
    role: str    # "user" | "assistant"
    content: str


@dataclass
class ConversationHistory:
    """
    單一使用者的對話歷史（環形緩衝）。

    maxlen 控制最多保留幾輪；超出後自動丟棄最舊的記錄，
    防止 context window 爆滿。
    """
    maxlen: int = 10
    turns: Deque[Turn] = field(default_factory=lambda: deque(maxlen=10))

    def add_user(self, text: str) -> None:
        self.turns.append(Turn(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self.turns.append(Turn(role="assistant", content=text))

    def clear(self) -> None:
        self.turns.clear()


# ─── In-memory 歷史倉儲（key = LINE user_id）─────────────────────────────────

_histories: dict[str, ConversationHistory] = {}


def get_history(user_id: str) -> ConversationHistory:
    """取得或建立指定使用者的對話歷史。"""
    if user_id not in _histories:
        _histories[user_id] = ConversationHistory()
    return _histories[user_id]


def clear_history(user_id: str) -> None:
    """清除指定使用者的對話歷史（用於「重設」指令）。"""
    if user_id in _histories:
        _histories[user_id].clear()
        logger.info(f"🗑️  已清除 {user_id} 的對話歷史。")


# ─── System Prompt ────────────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = """\
你是「Ē-Kóng (會講)」，一個具備情緒感知與台/中雙語能力的智慧陪伴 Agent。
你必須分析使用者的輸入（純文字），並以 JSON 格式輸出你的判斷與回應。

【意圖分類 (intent)】
- translate：使用者明確要求翻譯（例如：「幫我用台語說...」、「這句台語怎麼講」）
- companion：使用者在閒聊、訴苦、分享日常（例如：「今天老闆好煩」、「我好累」）

【情緒感知 (user_emotion)】判斷使用者的情緒狀態，可能值：neutral, sad, angry, happy, exhausted, anxious, lonely

【生成回應 (response_text)】
- 若 intent 為 translate：直接且精準地給出翻譯結果，無需多餘廢話。
- 若 intent 為 companion：展現極高的同理心與溫暖。如果是負面情緒，請先安撫與共情；
  語氣必須像是一個親切的台灣在地好鄰居，回覆 100 字以內（適合語音播放）。適時加入台灣常見語氣詞。

【response_language】
- "nan"：台語（使用者主要用台語或明確要求台語回覆）
- "zh-TW"：繁體中文（其餘情況）

【絕對禁止】
- 不可說「我是 AI」或揭露模型身份
- 不可在 JSON 之外輸出任何文字、解釋或 markdown

【輸出格式 — 只輸出此 JSON，不包含任何其他文字】
{
  "intent": "translate" | "companion",
  "user_emotion": "...",
  "response_language": "zh-TW" | "nan",
  "response_text": "你的具體回應內容"
}
"""


def build_prompt(
    user_id: str,
    user_input: str,
    emotion: Emotion,
) -> str:
    """
    組裝完整 ChatML 格式 Prompt。

    格式：
      <|im_start|>system\\n{system}\\n<|im_end|>
      <|im_start|>user\\n{msg}\\n<|im_end|>
      ...（歷史輪次）
      <|im_start|>assistant\\n

    Parameters
    ----------
    user_id : str
        LINE 使用者 ID（用於提取對話歷史）。
    user_input : str
        本輪使用者輸入文字。
    emotion : Emotion
        本輪偵測到的情緒（heuristic，提供給 prompt 額外上下文）。

    Returns
    -------
    str
        完整 prompt 字串，直接傳入 llm.generate()。
    """
    history = get_history(user_id)

    # system prompt（固定，emotion 上下文以 heuristic 偵測結果補充）
    system = _BASE_SYSTEM_PROMPT
    if emotion not in (Emotion.NEUTRAL,):
        system = system.rstrip() + (
            f"\n\n【本地情緒感知輔助：使用者可能感到 {emotion.zh} {emotion.emoji}，供參考。】\n"
        )

    parts: list[str] = [
        f"<|im_start|>system\n{system.strip()}<|im_end|>\n"
    ]

    # 加入歷史輪次（僅保留 response_text 部分避免污染格式）
    for turn in history.turns:
        parts.append(f"<|im_start|>{turn.role}\n{turn.content}<|im_end|>\n")

    # 加入本輪使用者輸入
    parts.append(f"<|im_start|>user\n{user_input}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")

    return "".join(parts)


# ─── JSON 解析 ────────────────────────────────────────────────────────────────

_JSON_EXTRACT_RE = re.compile(r"\{.*?\}", re.DOTALL)
# 支援 LLM 把 JSON 包在 ```json ... ``` 的情況
_MD_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_llm_json(raw: str, fallback_emotion: Emotion) -> AgentResponse:
    """
    解析 LLM 輸出的 JSON 字串，容錯處理各種邊界情況。

    容錯順序：
      1. 空字串 → 直接 fallback（避免 JSONDecodeError 噪音）
      2. Markdown code block 包裝（```json {...} ```）→ 抽取
      3. 裸 JSON 區塊（regex {…}）→ 抽取
      4. 整段當 json_str → 嘗試解析
      5. 全失敗 → fallback companion 回覆
    """
    if not raw.strip():
        logger.warning("⚠️  LLM 輸出空字串，直接使用 fallback 回覆。")
        return AgentResponse(
            intent=Intent.COMPANION,
            user_emotion=fallback_emotion.value,
            response_language="zh-TW",
            response_text="哎，我剛才沒反應過來，能再說一次嗎？",
            emotion=fallback_emotion,
        )

    # 嘗試從 markdown code block 抽取
    md_m = _MD_CODE_BLOCK_RE.search(raw)
    if md_m:
        json_str = md_m.group(1)
    else:
        # 嘗試從輸出中抽取最外層 JSON 區塊
        brace_m = _JSON_EXTRACT_RE.search(raw)
        json_str = brace_m.group(0) if brace_m else raw

    try:
        data = json.loads(json_str)
        intent_val = data.get("intent", "companion")
        intent = Intent(intent_val) if intent_val in Intent._value2member_map_ else Intent.COMPANION

        llm_emotion_str = data.get("user_emotion", fallback_emotion.value)
        try:
            resolved_emotion = Emotion(llm_emotion_str)
        except ValueError:
            resolved_emotion = fallback_emotion

        return AgentResponse(
            intent=intent,
            user_emotion=llm_emotion_str,
            response_language=data.get("response_language", "zh-TW"),
            response_text=data.get("response_text", raw).strip(),
            emotion=resolved_emotion,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning(
            f"⚠️  LLM JSON 解析失敗（{exc}）｜raw={raw!r:.120}，使用 fallback 回覆。"
        )
        return AgentResponse(
            intent=Intent.COMPANION,
            user_emotion=fallback_emotion.value,
            response_language="zh-TW",
            response_text=raw.strip() or "哎，我剛才沒反應過來，能再說一次嗎？",
            emotion=fallback_emotion,
        )


# ─── Agent 主流程 ─────────────────────────────────────────────────────────────

async def chat(user_id: str, user_input: str) -> tuple[str, Emotion, str]:
    """
    完整 Agent 推論流程：情緒偵測 → Prompt 組裝 → LLM 推論（JSON）→ 路由 → 更新歷史。

    Parameters
    ----------
    user_id : str
        LINE 使用者 ID。
    user_input : str
        使用者輸入文字（文字訊息或 STT 轉錄結果）。

    Returns
    -------
    tuple[str, Emotion, str]
        (最終回覆文字, 解析後的情緒, response_language "zh-TW"|"nan")

    Raises
    ------
    RuntimeError
        LLM 模型尚未初始化。
    """
    # 1. 本地 heuristic 情緒偵測（作為 LLM 輔助上下文 & fallback）
    emotion = detect_emotion(user_input)

    # 2. 組裝 Prompt
    prompt = build_prompt(user_id, user_input, emotion)

    logger.debug(
        f"💬 Agent｜{user_id}｜heuristic情緒={emotion.zh}｜"
        f"歷史輪數={len(get_history(user_id).turns)}"
    )

    # 3. LLM 推論（期望輸出 JSON）
    raw_output = await generate(prompt)

    # 4. 解析 JSON 並路由
    agent_resp = parse_llm_json(raw_output, fallback_emotion=emotion)

    logger.info(
        f"🎯 路由｜intent={agent_resp.intent.value}｜"
        f"emotion={agent_resp.user_emotion}｜lang={agent_resp.response_language}｜"
        f"{agent_resp.response_text[:60]}"
    )

    # 5. 更新對話歷史（存 response_text 而非整個 JSON，避免污染 next-turn context）
    history = get_history(user_id)
    history.add_user(user_input)
    history.add_assistant(agent_resp.response_text)

    return agent_resp.response_text, agent_resp.emotion, agent_resp.response_language


# ─── 指令解析（簡易規則型） ──────────────────────────────────────────────────────

_RESET_KEYWORDS = {"重設", "重置", "清除記憶", "忘掉之前", "forget", "reset"}


def parse_command(text: str) -> str | None:
    """
    解析使用者「特殊指令」文字，回傳指令名稱或 None。

    目前支援：
      - "重設" / "reset" 等 → "reset"

    Parameters
    ----------
    text : str
        使用者輸入文字。

    Returns
    -------
    str | None
        指令名稱（"reset"），或 None 表示普通對話。
    """
    normalized = text.strip().lower()
    for kw in _RESET_KEYWORDS:
        if kw in normalized:
            return "reset"
    return None
