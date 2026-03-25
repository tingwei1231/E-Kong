"""
app/services/agent.py — 情緒感知 LLM Agent 服務
=================================================
職責：
  1. 對話歷史管理（In-memory，per-user，環形緩衝）
  2. 情緒偵測（關鍵詞 + 標點啟發式規則，輕量快速）
  3. 系統 Prompt 組裝（含角色設定 + 情緒上下文 + 對話歷史）
  4. 呼叫 llm.generate() 取得回覆

設計原則：
  - 歷史記錄以 deque(maxlen=N) 實作環形緩衝，防記憶體無限增長
  - 情緒偵測與 Prompt 組裝分離，便於後續替換為 BERT 情緒分類模型
  - TAIDE / Llama-3 均使用 ChatML 格式（<|im_start|> … <|im_end|>）

Prompt 格式（ChatML）：
  <|im_start|>system
  {system_prompt}<|im_end|>
  <|im_start|>user
  {user_msg}<|im_end|>
  <|im_start|>assistant
"""

from __future__ import annotations

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
    HAPPY    = "happy"
    SAD      = "sad"
    ANXIOUS  = "anxious"
    ANGRY    = "angry"
    LONELY   = "lonely"
    NEUTRAL  = "neutral"

    @property
    def zh(self) -> str:
        """中文標籤（用於 Prompt 組裝）。"""
        return {
            "happy":   "開心",
            "sad":     "難過",
            "anxious": "焦慮",
            "angry":   "生氣",
            "lonely":  "孤獨",
            "neutral": "平靜",
        }[self.value]

    @property
    def emoji(self) -> str:
        return {
            "happy": "😊", "sad": "😢", "anxious": "😰",
            "angry": "😤", "lonely": "🥺", "neutral": "😌",
        }[self.value]


# ─── 情緒偵測（輕量 heuristic，Step 4 後可替換 BERT 分類器）───────────────────

# 關鍵詞對應情緒類別（中文 + 台語羅馬拼音常見詞）
_EMOTION_KEYWORDS: dict[Emotion, list[str]] = {
    Emotion.HAPPY:   ["開心", "高興", "快樂", "棒", "太好了", "哈哈", "😄", "🎉", "爽", "讚"],
    Emotion.SAD:     ["難過", "傷心", "哭", "可憐", "失落", "沮喪", "心痛", "痛苦", "😢", "😭"],
    Emotion.ANXIOUS: ["擔心", "焦慮", "緊張", "不安", "怕", "害怕", "恐慌", "壓力", "😰", "😨"],
    Emotion.ANGRY:   ["氣", "生氣", "憤怒", "煩", "討厭", "幹", "靠", "怒", "😤", "😠"],
    Emotion.LONELY:  ["寂寞", "孤獨", "孤單", "沒人", "沒朋友", "想你", "思念", "🥺", "😔"],
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
你是「Ē-Kóng（會講）」，一個溫柔、有耐心、善解人意的語音陪伴助手。
你擅長中文與台語，能自然切換兩種語言回應使用者。

回覆原則：
- 語氣親切溫暖，像老朋友在聊天
- 回覆簡短扼要（100 字以內），適合語音播放
- 不說教、不評判，以同理心回應情緒
- 若使用者說台語，盡量用台語或中台混用回覆
- 絕對不說「我是 AI」或揭露自己的模型身份
"""

_EMOTION_CONTEXT_TEMPLATE = "【當前情緒感知：使用者似乎感到{emotion_zh} {emoji}，請以同理心回應。】\n"


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
        本輪偵測到的情緒。

    Returns
    -------
    str
        完整 prompt 字串，直接傳入 llm.generate()。
    """
    history = get_history(user_id)

    # 組裝 system prompt
    system = _BASE_SYSTEM_PROMPT
    if emotion != Emotion.NEUTRAL:
        system += _EMOTION_CONTEXT_TEMPLATE.format(
            emotion_zh=emotion.zh, emoji=emotion.emoji
        )

    parts: list[str] = [
        f"<|im_start|>system\n{system.strip()}<|im_end|>\n"
    ]

    # 加入歷史輪次
    for turn in history.turns:
        parts.append(f"<|im_start|>{turn.role}\n{turn.content}<|im_end|>\n")

    # 加入本輪使用者輸入
    parts.append(f"<|im_start|>user\n{user_input}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")

    return "".join(parts)


# ─── Agent 主流程 ─────────────────────────────────────────────────────────────

async def chat(user_id: str, user_input: str) -> tuple[str, Emotion]:
    """
    完整 Agent 推論流程：情緒偵測 → Prompt 組裝 → LLM 推論 → 更新歷史。

    Parameters
    ----------
    user_id : str
        LINE 使用者 ID。
    user_input : str
        使用者輸入文字（文字訊息或 STT 轉錄結果）。

    Returns
    -------
    tuple[str, Emotion]
        (LLM 回覆文字, 偵測到的情緒)

    Raises
    ------
    RuntimeError
        LLM 模型尚未初始化。
    """
    # 1. 情緒偵測
    emotion = detect_emotion(user_input)

    # 2. 組裝 Prompt
    prompt = build_prompt(user_id, user_input, emotion)

    logger.debug(
        f"💬 Agent｜{user_id}｜情緒={emotion.zh}｜"
        f"歷史輪數={len(get_history(user_id).turns)}"
    )

    # 3. LLM 推論
    reply = await generate(prompt)

    # 4. 更新對話歷史
    history = get_history(user_id)
    history.add_user(user_input)
    history.add_assistant(reply)

    return reply, emotion


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
