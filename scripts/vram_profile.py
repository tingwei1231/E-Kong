"""
scripts/vram_profile.py — VRAM 使用量量測與報告
=================================================
在 Colab 中直接執行：
  !python scripts/vram_profile.py

功能：
  1. 載入各模型前後量測 GPU VRAM 佔用差異
  2. 輸出清晰的表格報告（含累計使用量與安全閾值檢查）
  3. 提供「輕量模式」建議（VRAM 超標時自動建議調整參數）

使用方式：
  # 全量測試（依序載入所有模型，耗時約 3–5 分鐘）
  python scripts/vram_profile.py

  # 只測特定模型
  python scripts/vram_profile.py --only stt llm
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# 確保可以 import app 模組
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── 資料結構 ──────────────────────────────────────────────────────────────────

@dataclass
class ModelProfile:
    """單一模型的 VRAM 測量結果。"""
    name: str
    vram_before_mb: float = 0.0
    vram_after_mb: float = 0.0
    load_time_sec: float = 0.0
    error: str | None = None

    @property
    def vram_used_mb(self) -> float:
        return max(0.0, self.vram_after_mb - self.vram_before_mb)

    @property
    def status(self) -> str:
        if self.error:
            return "❌ 失敗"
        return "✅ 成功"


# ─── GPU 量測工具 ─────────────────────────────────────────────────────────────

def get_gpu_vram_mb() -> float:
    """
    取得目前 GPU 已分配 VRAM（MB）。

    Returns
    -------
    float
        VRAM 使用量（MB）；無 GPU 時回傳 0.0。
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        # allocated = 實際使用量（不含 PyTorch cache）
        return torch.cuda.memory_allocated() / 1024 ** 2
    except ImportError:
        return 0.0


def get_total_vram_mb() -> float:
    """取得 GPU 總 VRAM（MB）；無 GPU 時回傳 0。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    except ImportError:
        return 0.0


def clear_gpu_cache() -> None:
    """清除 PyTorch GPU cache（不影響已分配的 VRAM）。"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ─── 模型量測流程 ─────────────────────────────────────────────────────────────

async def profile_stt() -> ModelProfile:
    """量測 Whisper STT VRAM 佔用。"""
    profile = ModelProfile(name="Whisper STT")
    try:
        from app.models.stt import close_stt, init_stt

        clear_gpu_cache()
        profile.vram_before_mb = get_gpu_vram_mb()
        t0 = time.perf_counter()

        await init_stt()

        profile.load_time_sec = time.perf_counter() - t0
        profile.vram_after_mb = get_gpu_vram_mb()

        await close_stt()
    except Exception as exc:  # noqa: BLE001
        profile.error = str(exc)
    return profile


async def profile_llm() -> ModelProfile:
    """量測 LLM VRAM 佔用。"""
    profile = ModelProfile(name="LLM (GGUF)")
    try:
        from app.models.llm import close_llm, init_llm

        clear_gpu_cache()
        profile.vram_before_mb = get_gpu_vram_mb()
        t0 = time.perf_counter()

        await init_llm()

        profile.load_time_sec = time.perf_counter() - t0
        profile.vram_after_mb = get_gpu_vram_mb()

        await close_llm()
    except Exception as exc:  # noqa: BLE001
        profile.error = str(exc)
    return profile


async def profile_tts_zh() -> ModelProfile:
    """量測 ChatTTS 中文 TTS VRAM 佔用。"""
    profile = ModelProfile(name="ChatTTS (中文)")
    try:
        from app.models.tts_zh import close_tts_zh, init_tts_zh

        clear_gpu_cache()
        profile.vram_before_mb = get_gpu_vram_mb()
        t0 = time.perf_counter()

        await init_tts_zh()

        profile.load_time_sec = time.perf_counter() - t0
        profile.vram_after_mb = get_gpu_vram_mb()

        await close_tts_zh()
    except Exception as exc:  # noqa: BLE001
        profile.error = str(exc)
    return profile


async def profile_tts_tw() -> ModelProfile:
    """量測 MMS-TTS 台語 VRAM 佔用。"""
    profile = ModelProfile(name="MMS-TTS (台語)")
    try:
        from app.models.tts_tw import close_tts_tw, init_tts_tw

        clear_gpu_cache()
        profile.vram_before_mb = get_gpu_vram_mb()
        t0 = time.perf_counter()

        await init_tts_tw()

        profile.load_time_sec = time.perf_counter() - t0
        profile.vram_after_mb = get_gpu_vram_mb()

        await close_tts_tw()
    except Exception as exc:  # noqa: BLE001
        profile.error = str(exc)
    return profile


# ─── 報告輸出 ─────────────────────────────────────────────────────────────────

_VRAM_LIMIT_MB = 15 * 1024  # T4 15 GB
_SAFETY_RATIO  = 0.85        # 建議使用上限 85%（保留 overhead）

def print_report(profiles: list[ModelProfile]) -> None:
    """輸出 VRAM 量測報告表格。"""
    total_vram = get_total_vram_mb()
    limit_mb   = total_vram if total_vram > 0 else _VRAM_LIMIT_MB
    safe_mb    = limit_mb * _SAFETY_RATIO

    print("\n" + "═" * 70)
    print("  Project Ē-Kóng — VRAM 量測報告")
    print(f"  GPU 總 VRAM：{limit_mb / 1024:.1f} GB   安全上限：{safe_mb / 1024:.1f} GB")
    print("═" * 70)

    header = f"{'模型':<20} {'VRAM 佔用':>10} {'載入時間':>10} {'狀態':>8}"
    print(header)
    print("─" * 70)

    cumulative = 0.0
    for p in profiles:
        if p.error:
            print(f"{p.name:<20} {'N/A':>10} {'N/A':>10} {p.status:>8}  ({p.error[:30]})")
        else:
            cumulative += p.vram_used_mb
            print(
                f"{p.name:<20}"
                f" {p.vram_used_mb / 1024:>8.2f} GB"
                f" {p.load_time_sec:>8.1f}s"
                f" {p.status:>8}"
            )

    print("─" * 70)
    ratio = cumulative / limit_mb if limit_mb > 0 else 0
    flag  = "✅ 安全" if cumulative <= safe_mb else "⚠️  超標"
    print(f"{'合計 VRAM':<20} {cumulative / 1024:>8.2f} GB {'':>10} {flag:>8}")
    print(f"{'GPU 使用率':<20} {ratio * 100:>7.1f}%")
    print("═" * 70)

    # 超標建議
    if cumulative > safe_mb:
        print("\n⚠️  VRAM 使用超過安全閾值，建議調整：")
        print("   1. 降低 LLM_N_GPU_LAYERS（例：從 35 → 25，部分層移至 CPU）")
        print("   2. 改用更小的 Whisper 模型（base → tiny）")
        print("   3. 延遲載入 TTS（首次使用時才初始化）")
        print("   4. 同時間只保留 STT + LLM，TTS 按需載入後卸載")
    print()


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

_PROFILERS = {
    "stt": profile_stt,
    "llm": profile_llm,
    "tts_zh": profile_tts_zh,
    "tts_tw": profile_tts_tw,
}


async def main(only: list[str] | None = None) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    tasks = only or list(_PROFILERS.keys())
    print(f"\n🔍 量測模組：{', '.join(tasks)}")

    profiles: list[ModelProfile] = []
    for key in tasks:
        if key not in _PROFILERS:
            print(f"⚠️  未知模組：{key}，跳過。")
            continue
        print(f"  ⏳ 正在量測 {key}…")
        p = await _PROFILERS[key]()
        profiles.append(p)

    print_report(profiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ē-Kóng VRAM Profiler")
    parser.add_argument(
        "--only", nargs="+",
        choices=list(_PROFILERS.keys()),
        help="只量測指定模組（stt / llm / tts_zh / tts_tw）",
    )
    args = parser.parse_args()
    asyncio.run(main(only=args.only))
