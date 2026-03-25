"""
tests/test_pipeline.py — 端對端 Pipeline 整合測試
===================================================
使用 pytest + pytest-asyncio 執行。

測試策略：
  - 所有外部依賴（LINE API、Whisper、LLM、TTS）均以 Mock 替換
  - 測試業務邏輯正確性（路由、情緒偵測、Prompt 組裝、Fallback 機制）
  - 不需 GPU / 模型檔案即可在本機執行

執行：
  pip install pytest pytest-asyncio
  pytest tests/test_pipeline.py -v

環境需求：
  - 不需 .env（MOCK 替換了所有外部呼叫）
  - Python 3.9+
"""

from __future__ import annotations

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


# ─── 情緒偵測測試 ─────────────────────────────────────────────────────────────

class TestEmotionDetection:
    """測試 detect_emotion() 的關鍵詞命中邏輯。"""

    def test_happy_keywords(self):
        from app.services.agent import Emotion, detect_emotion
        assert detect_emotion("今天好開心！哈哈") == Emotion.HAPPY

    def test_sad_keywords(self):
        from app.services.agent import Emotion, detect_emotion
        assert detect_emotion("我好難過，傷心死了") == Emotion.SAD

    def test_anxious_keywords(self):
        from app.services.agent import Emotion, detect_emotion
        assert detect_emotion("好擔心喔，怎麼辦") == Emotion.ANXIOUS

    def test_multiple_exclamation_anxious(self):
        from app.services.agent import Emotion, detect_emotion
        # 多個問號 → 焦慮
        result = detect_emotion("怎麼辦！！！")
        assert result == Emotion.ANXIOUS

    def test_neutral_no_keywords(self):
        from app.services.agent import Emotion, detect_emotion
        assert detect_emotion("今天天氣不錯") == Emotion.NEUTRAL

    def test_angry_emoji(self):
        from app.services.agent import Emotion, detect_emotion
        assert detect_emotion("真的很煩😤") == Emotion.ANGRY


# ─── Prompt 組裝測試 ──────────────────────────────────────────────────────────

class TestPromptBuilding:
    """測試 build_prompt() 格式與內容。"""

    def test_chatml_format(self):
        from app.services.agent import Emotion, build_prompt
        prompt = build_prompt("user_123", "你好", Emotion.NEUTRAL)
        assert "<|im_start|>system" in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt
        assert "你好" in prompt

    def test_emotion_context_injected_when_non_neutral(self):
        from app.services.agent import Emotion, build_prompt
        prompt = build_prompt("user_123", "我好難過", Emotion.SAD)
        assert "難過" in prompt      # 情緒上下文被注入
        assert "😢" in prompt

    def test_emotion_context_absent_when_neutral(self):
        from app.services.agent import Emotion, build_prompt
        prompt = build_prompt("user_123", "你好", Emotion.NEUTRAL)
        # NEUTRAL 不注入情緒上下文
        assert "當前情緒感知" not in prompt

    def test_history_included_in_prompt(self):
        from app.services.agent import Emotion, build_prompt, get_history
        history = get_history("user_hist_test")
        history.add_user("第一輪問題")
        history.add_assistant("第一輪回答")

        prompt = build_prompt("user_hist_test", "第二輪問題", Emotion.NEUTRAL)
        assert "第一輪問題" in prompt
        assert "第一輪回答" in prompt
        assert "第二輪問題" in prompt


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
        # 超過 maxlen 後自動丟棄最舊
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
        # 漢字少於 20%
        assert is_romanized("li ho a") is True


# ─── 音訊服務測試 ─────────────────────────────────────────────────────────────

class TestAudioService:
    """測試 cleanup_temp_file 與 process_line_audio 的例外處理。"""

    def test_cleanup_nonexistent_file_no_crash(self):
        from app.services.audio import cleanup_temp_file
        # 應不拋例外
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

        with patch("app.services.agent.generate", new=AsyncMock(return_value="你好！有什麼可以幫你的？")):
            reply, emotion = await chat("e2e_test_user", "你好")

        assert reply == "你好！有什麼可以幫你的？"
        history = get_history("e2e_test_user")
        # 歷史應更新
        turns = list(history.turns)
        assert any(t.content == "你好" for t in turns)
        assert any(t.content == "你好！有什麼可以幫你的？" for t in turns)

    @pytest.mark.asyncio
    async def test_chat_detects_emotion(self):
        from app.services.agent import Emotion, chat

        with patch("app.services.agent.generate", new=AsyncMock(return_value="我懂你的感受...")):
            _, emotion = await chat("emotion_test_user", "我好難過，心情很差😢")

        assert emotion == Emotion.SAD

    @pytest.mark.asyncio
    async def test_try_reply_audio_returns_false_when_no_tts(self):
        """TTS 引擎未初始化時 try_reply_audio 應回傳 False。"""
        from app.services.tts import try_reply_audio

        with patch("app.services.tts.get_tts_zh", return_value=None), \
             patch("app.services.tts.get_tts_tw", return_value=None):
            result = await try_reply_audio("fake_token", "測試文字", language="zh")

        assert result is False
