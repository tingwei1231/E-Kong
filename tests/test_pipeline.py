"""
tests/test_pipeline.py — 端對端 Pipeline 整合測試
===================================================
使用 pytest + pytest-asyncio 執行。

測試策略：
  - 所有外部依賴（LINE API、Whisper、LLM、TTS）均以 Mock 替換
  - 測試業務邏輯正確性（intent 路由、Prompt 組裝、JSON 解析、Fallback 機制）
  - 不需 GPU / 模型檔案即可在本機執行

執行：
  pip install pytest pytest-asyncio
  pytest tests/test_pipeline.py -v

環境需求：
  - 不需 .env（MOCK 替換了所有外部呼叫）
  - Python 3.9+
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 確保 import 路徑正確
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_settings():
    """Mock pydantic Settings，避免需要真實 .env。"""
    fake = MagicMock()
    fake.line_channel_access_token = "FAKE_TOKEN"
    fake.line_channel_secret = "FAKE_SECRET"
    fake.ngrok_auth_token = "FAKE_NGROK"
    fake.llm_model_path = "/tmp/fake.gguf"
    fake.llm_n_gpu_layers = 0
    fake.llm_max_tokens = 256
    fake.llm_temperature = 0.7
    fake.whisper_model_size = "tiny"
    fake.whisper_device = "cpu"
    fake.whisper_compute_type = "int8"
    fake.log_level = "DEBUG"
    fake.app_host = "0.0.0.0"
    fake.app_port = 8000

    with patch("app.config.get_settings", return_value=fake):
        yield fake


# ─── JSON 解析測試 ────────────────────────────────────────────────────────────

class TestParseJson:
    """測試 parse_llm_json() 的正確性與容錯能力。"""

    def test_update_score_intent(self):
        from app.services.agent import Intent, parse_llm_json
        raw = json.dumps({
            "intent": "Update_Score",
            "response_text": "已更新比分。",
            "action": {"match_id": "A組第一場", "team_a": "台大", "score_a": 25,
                       "team_b": "政大", "score_b": 18, "winner": "台大"},
        })
        resp = parse_llm_json(raw)
        assert resp.intent == Intent.UPDATE_SCORE
        assert resp.response_text == "已更新比分。"
        assert resp.action["winner"] == "台大"

    def test_broadcast_intent(self):
        from app.services.agent import Intent, parse_llm_json
        raw = json.dumps({
            "intent": "Broadcast",
            "response_text": "廣播已送出。",
            "action": {"target": "all", "message": "請裁判到主場地集合。"},
        })
        resp = parse_llm_json(raw)
        assert resp.intent == Intent.BROADCAST
        assert resp.action["target"] == "all"

    def test_general_chat_action_is_none(self):
        from app.services.agent import Intent, parse_llm_json
        raw = json.dumps({
            "intent": "General_Chat",
            "response_text": "決賽在三點開始。",
            "action": None,
        })
        resp = parse_llm_json(raw)
        assert resp.intent == Intent.GENERAL_CHAT
        assert resp.action is None

    def test_unknown_intent_fallback_to_general_chat(self):
        from app.services.agent import Intent, parse_llm_json
        raw = json.dumps({
            "intent": "random_unknown",
            "response_text": "??",
            "action": None,
        })
        resp = parse_llm_json(raw)
        assert resp.intent == Intent.GENERAL_CHAT

    def test_empty_string_fallback(self):
        from app.services.agent import Intent, parse_llm_json
        resp = parse_llm_json("")
        assert resp.intent == Intent.GENERAL_CHAT
        assert resp.action is None

    def test_markdown_code_block_extracted(self):
        from app.services.agent import Intent, parse_llm_json
        raw = '```json\n{"intent": "General_Chat", "response_text": "好的。", "action": null}\n```'
        resp = parse_llm_json(raw)
        assert resp.intent == Intent.GENERAL_CHAT
        assert resp.response_text == "好的。"

    def test_malformed_json_fallback(self):
        from app.services.agent import Intent, parse_llm_json
        resp = parse_llm_json("這不是 JSON 格式的東西")
        assert resp.intent == Intent.GENERAL_CHAT


# ─── Prompt 組裝測試 ──────────────────────────────────────────────────────────

class TestPromptBuilding:
    """測試 build_prompt() 格式與內容。"""

    def test_chatml_format(self):
        from app.services.agent import build_prompt
        prompt = build_prompt("user_123", "比賽幾點開始？")
        assert "<|im_start|>system" in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt
        assert "比賽幾點開始？" in prompt

    def test_history_included_in_prompt(self):
        from app.services.agent import build_prompt, get_history
        history = get_history("user_hist_test")
        history.add_user("第一輪問題")
        history.add_assistant("第一輪回答")

        prompt = build_prompt("user_hist_test", "第二輪問題")
        assert "第一輪問題" in prompt
        assert "第一輪回答" in prompt
        assert "第二輪問題" in prompt

    def test_system_prompt_contains_intents(self):
        from app.services.agent import build_prompt
        prompt = build_prompt("user_sys", "test")
        assert "Update_Score" in prompt
        assert "Broadcast" in prompt
        assert "General_Chat" in prompt


# ─── 對話歷史測試 ─────────────────────────────────────────────────────────────

class TestConversationHistory:
    """測試環形緩衝對話歷史。"""

    def test_clear_history(self):
        from app.services.agent import clear_history, get_history
        h = get_history("clear_test_user")
        h.add_user("msg1")
        clear_history("clear_test_user")
        assert len(get_history("clear_test_user").turns) == 0

    def test_ring_buffer_maxlen(self):
        from app.services.agent import ConversationHistory
        h = ConversationHistory(maxlen=4)
        h.turns = __import__("collections").deque(maxlen=4)
        for i in range(10):
            h.add_user(f"msg{i}")
        assert len(h.turns) == 4
        assert h.turns[-1].content == "msg9"

    def test_parse_reset_command(self):
        from app.services.agent import parse_command
        assert parse_command("重設") == "reset"
        assert parse_command("RESET") == "reset"
        assert parse_command("清除記憶") == "reset"
        assert parse_command("你好啊") is None


# ─── TTS 路由測試 ─────────────────────────────────────────────────────────────

class TestTTSRouting:
    """測試 TTS 語言路由邏輯。"""

    def test_picks_zh_for_zh(self):
        from app.services.tts import _pick_engine
        assert _pick_engine("zh") == "zh"

    def test_picks_tw_for_nan(self):
        from app.services.tts import _pick_engine
        assert _pick_engine("nan") == "tw"

    def test_picks_zh_for_none(self):
        from app.services.tts import _pick_engine
        assert _pick_engine(None) == "zh"

    def test_picks_zh_for_english(self):
        from app.services.tts import _pick_engine
        assert _pick_engine("en") == "zh"


# ─── TTS 台語拼音偵測測試 ─────────────────────────────────────────────────────

class TestRomanizedDetection:
    """測試 is_romanized() 的漢字比例偵測。"""

    def test_pure_hanzi_is_not_romanized(self):
        from app.models.tts_tw import is_romanized
        assert is_romanized("你好我是台灣人") is False

    def test_pure_latin_is_romanized(self):
        from app.models.tts_tw import is_romanized
        assert is_romanized("li ho, gua si taiwan lang") is True

    def test_mixed_mostly_latin_is_romanized(self):
        from app.models.tts_tw import is_romanized
        assert is_romanized("li ho a") is True


# ─── 音訊服務測試 ─────────────────────────────────────────────────────────────

class TestAudioService:
    """測試 cleanup_temp_file 與 process_line_audio 的例外處理。"""

    def test_cleanup_nonexistent_file_no_crash(self):
        from app.services.audio import cleanup_temp_file
        cleanup_temp_file("/tmp/nonexistent_ekong_test.wav")

    @pytest.mark.asyncio
    async def test_process_line_audio_http_error(self):
        """LINE Content API 回傳 4xx 時應拋出 HTTPStatusError。"""
        import httpx
        from app.services.audio import process_line_audio

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 401
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_resp
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_ctx

            with pytest.raises(httpx.HTTPStatusError):
                await process_line_audio("fake_message_id")


# ─── LLM Agent 端對端流程測試 ────────────────────────────────────────────────

class TestAgentChatFlow:
    """Mock LLM，測試完整 chat() 流程。"""

    @pytest.mark.asyncio
    async def test_chat_updates_history(self):
        from app.services.agent import chat, get_history

        llm_output = json.dumps({
            "intent": "General_Chat",
            "response_text": "決賽在三點開始，請準時入場。",
            "action": None,
        })
        with patch("app.services.agent.generate", new=AsyncMock(return_value=llm_output)):
            resp = await chat("e2e_test_user", "決賽幾點？")

        assert resp.response_text == "決賽在三點開始，請準時入場。"
        history = get_history("e2e_test_user")
        turns = list(history.turns)
        assert any(t.content == "決賽幾點？" for t in turns)
        assert any(t.content == "決賽在三點開始，請準時入場。" for t in turns)

    @pytest.mark.asyncio
    async def test_chat_update_score_intent(self):
        from app.services.agent import Intent, chat

        llm_output = json.dumps({
            "intent": "Update_Score",
            "response_text": "已更新台大對政大，台大 25:18 獲勝。",
            "action": {"match_id": "A組第一場", "team_a": "台大", "score_a": 25,
                       "team_b": "政大", "score_b": 18, "winner": "台大"},
        })
        with patch("app.services.agent.generate", new=AsyncMock(return_value=llm_output)):
            resp = await chat("score_test_user", "A組第一場台大對政大25比18台大勝")

        assert resp.intent == Intent.UPDATE_SCORE
        assert resp.action is not None
        assert resp.action["winner"] == "台大"

    @pytest.mark.asyncio
    async def test_chat_broadcast_intent(self):
        from app.services.agent import Intent, chat

        llm_output = json.dumps({
            "intent": "Broadcast",
            "response_text": "廣播已送出。",
            "action": {"target": "all", "message": "請裁判到主場地集合。"},
        })
        with patch("app.services.agent.generate", new=AsyncMock(return_value=llm_output)):
            resp = await chat("broadcast_test_user", "廣播請裁判到主場地集合")

        assert resp.intent == Intent.BROADCAST
        assert resp.action["target"] == "all"

    @pytest.mark.asyncio
    async def test_try_reply_audio_returns_false_when_no_tts(self):
        """TTS 引擎未初始化時 try_reply_audio 應回傳 False。"""
        from app.services.tts import try_reply_audio

        with patch("app.services.tts.get_tts_zh", return_value=None), \
             patch("app.services.tts.get_tts_tw", return_value=None):
            result = await try_reply_audio("fake_token", "測試文字", language="zh")

        assert result is False
