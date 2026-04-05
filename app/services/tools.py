"""
app/services/tools.py — 外部工具函式庫
======================================
目前工具：
  - tool_query_google_sheet(match_id): 透過 Service Account 讀取 Google Sheets
    取得指定場次的即時比分與狀態，並套用 N+1 場次顯示邏輯。

N+1 顯示邏輯說明：
  - Google Sheets 中場次以 0-based index 儲存（第 0 列 = 賽事第一場）
  - 現場工作人員與裁判的習慣稱呼是「第一場」「第二場」…
  - 因此回傳給 LLM 的場次標籤皆為 sheet_row_index + 1
  - 例：試算表第 0 列 → 顯示為「第 1 場」

Service Account 設定：
  - GCP Console 建立 Service Account，下載 JSON 金鑰
  - 將金鑰路徑寫入 .env：GOOGLE_SERVICE_ACCOUNT_JSON=/path/to/key.json
  - 將金鑰放於 Google Drive：/content/drive/MyDrive/secrets/sa_key.json（Colab 用）
  - 將試算表 ID 寫入 .env：GOOGLE_SHEET_ID=<spreadsheet_id>
  - 在 Google Sheets 頁面右上角「共用」，將 Service Account email 加入（至少檢視者）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from loguru import logger

# ─── 依賴（Python 內建）──────────────────────────────────────────────────────────
import csv
import urllib.request
import ssl

# 試算表欄位定義（A~K，共 11 欄）
_COL_PHASE     = 0   # A：賽段（預賽 / 複賽）
_COL_MATCH_ID  = 1   # B：場次（e.g. "A組1場"）
_COL_TIME      = 2   # C：時段（e.g. "8:30"）
_COL_COURT     = 3   # D：場地（A / B / C）
_COL_TEAM_A    = 4   # E：隊伍 A
_COL_TEAM_B    = 5   # F：隊伍 B
_COL_SCORE_A   = 6   # G：比分 A（局數 0/1/2）
_COL_SCORE_B   = 7   # H：比分 B（局數 0/1/2）
_COL_SET_SCORE = 8   # I：局分（e.g. "25:18,25:19"）
_COL_STATUS    = 9   # J：狀態（已結束 / 未開始 / 進行中）
_COL_WINNER    = 10  # K：獲勝隊伍
_COL_WORKSHEET = "賽程"  # 工作表分頁名稱


# ─── 回傳資料結構 ──────────────────────────────────────────────────────────────

@dataclass
class MatchQueryResult:
    """Google Sheets 查詢結果。"""
    found:         bool
    match_id:      str
    display_label: str
    phase:         str              # 賽段
    time:          str              # 時段
    court:         str              # 場地
    team_a:        str
    team_b:        str
    score_a:       int | str        # 局數（未開始時可能是 "-"）
    score_b:       int | str
    set_score:     str              # 多局局分（e.g. "25:18,25:19"）
    status:        str
    winner:        str | None
    error_message: str | None = None


# ─── CSV 下載與解析核心 ────────────────────────────────────────────────────────
# 使用 urllib 抓取 Google Sheets 發布的 CSV 網址

def _fetch_csv_rows_by_url(url: str) -> list[list[str]]:
    """下載單一 CSV URL 並解析成二維陣列（不含表頭）。"""    
    try:
        # 忽略 Colab/本地 可能的 SSL 憑證問題
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as response:
            content = response.read().decode('utf-8')
            
        reader = csv.reader(content.splitlines())
        rows = list(reader)
        return rows[1:] if rows else []  # 跳過第一列表頭
    except Exception as e:
        logger.error(f"下載或解析 CSV ({url}) 失敗：{e}")
        return []

def _fetch_csv_rows(env_key: str) -> list[list[str]]:
    """從指定環境變數取得 CSV URL，下載並解析成二維陣列（不含表頭）。"""
    url = os.getenv(env_key)
    if not url:
        raise RuntimeError(f"環境變數 {env_key} 未設定，請填入發布的 CSV 網址。")
    return _fetch_csv_rows_by_url(url.strip())


# ─── N+1 場次標籤轉換 ─────────────────────────────────────────────────────────

def _to_display_label(row_index: int) -> str:
    """
    將試算表 0-based row index 轉換為現場人員習慣的場次標籤。

    試算表第 0 列（第一筆資料）→ 「第 1 場」
    試算表第 1 列 → 「第 2 場」
    以此類推（N+1）
    """
    return f"第 {row_index + 1} 場"


# ─── 主要查詢函式 ──────────────────────────────────────────────────────────────

def tool_query_google_sheet(match_id: str) -> MatchQueryResult:
    """
    查詢 Google Sheets 中指定場次的即時比分與狀態。
    """
    try:
        data_rows = _fetch_csv_rows("GOOGLE_SHEET_CSV_SCORE")
        
        normalized_query = match_id.strip().lower()
        
        for idx, row in enumerate(data_rows):
            if len(row) <= _COL_WINNER:
                continue  # 跳過欄位不足的列

            cell_match_id = row[_COL_MATCH_ID].strip().lower()
            if cell_match_id != normalized_query:
                continue

            # ── 找到對應場次 ────────────────────────────────────────────────
            display_label = _to_display_label(idx)  # N+1

            raw_score_a = row[_COL_SCORE_A].strip()
            raw_score_b = row[_COL_SCORE_B].strip()

            return MatchQueryResult(
                found=True,
                match_id=row[_COL_MATCH_ID].strip(),
                display_label=display_label,
                phase=row[_COL_PHASE].strip(),
                time=row[_COL_TIME].strip(),
                court=row[_COL_COURT].strip(),
                team_a=row[_COL_TEAM_A].strip(),
                team_b=row[_COL_TEAM_B].strip(),
                score_a=int(raw_score_a) if raw_score_a.isdigit() else raw_score_a,
                score_b=int(raw_score_b) if raw_score_b.isdigit() else raw_score_b,
                set_score=row[_COL_SET_SCORE].strip() if len(row) > _COL_SET_SCORE else "",
                status=row[_COL_STATUS].strip(),
                winner=row[_COL_WINNER].strip() or None,
            )

        # 找不到
        return MatchQueryResult(
            found=False, match_id=match_id, display_label="", phase="", time="", court="", team_a="", score_a=0,
            team_b="", score_b=0, set_score="", status="", winner=None,
            error_message=f"找不到場次 {match_id!r}，請確認場次 ID 是否正確。",
        )

    except Exception as exc:  # noqa: BLE001
        logger.error(f"Google Sheets CSV 查詢失敗：{exc}")
        return MatchQueryResult(
            found=False, match_id=match_id, display_label="", phase="", time="", court="", team_a="", score_a=0,
            team_b="", score_b=0, set_score="", status="", winner=None,
            error_message=f"讀取比分資料失敗，請確認 CSV 網址設定。（{exc}）",
        )


def format_match_result_for_llm(result: MatchQueryResult) -> str:
    """
    將 MatchQueryResult 格式化為 LLM 可直接使用的結構化文字，
    作為 System Prompt 的額外工具輸出上下文。
    """
    if not result.found:
        return f"[工具查詢結果] 查無資料：{result.error_message}"

    winner_str = f"獲勝隊伍：{result.winner}" if result.winner else "獲勝隊伍：尚未決定"
    set_str    = f"局分明細：{result.set_score}\n" if result.set_score else ""
    return (
        f"[即時比分查詢結果]\n"
        f"賽段：{result.phase}｜{result.display_label}（ID: {result.match_id}）｜場地：{result.court}｜時段：{result.time}\n"
        f"{result.team_a} {result.score_a}局 : {result.score_b}局 {result.team_b}（互局制）\n"
        f"{set_str}"
        f"狀態：{result.status}｜{winner_str}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 動態賽程查詢工具（獨立分頁，與比分 worksheet 分開）
#
# 資料來源為 5 個發布為 CSV 的不同網址，對應的基礎環境變數如下：
#   - GOOGLE_SHEET_CSV_SCORE       ← 比分（上方工具）
#   - GOOGLE_SHEET_CSV_LOSER_STANDINGS ← 敗者組積分表
#   - GOOGLE_SHEET_CSV_GROUPS      ← 各組隊伍名單
#   - GOOGLE_SHEET_CSV_STANDINGS   ← 各組積分排名
#   - GOOGLE_SHEET_CSV_ELIMINATION ← 晉級與淘汰結果
#
# ═══════════════════════════════════════════════════════════════════════════════

# ─── 1. 敗者組積分查詢 ─────────────────────────────────────────────────────────────
#
# 敗者組積分分頁欄位（A~E）：
#   A：組別  B：隊伍名稱  C：平均勝敗局  D：平均全失分  E：狀態
#
_LOS_GROUP        = 0
_LOS_TEAM         = 1
_LOS_AVG_SETS     = 2
_LOS_AVG_LOST_PTS = 3
_LOS_STATUS       = 4


def tool_query_loser_standings(group_name: str | None = None) -> str:
    """
    查詢敗者組積分。

    Parameters
    ----------
    group_name : str | None
        "A組" / "B組" / None（全部）

    Returns
    -------
    str
        格式化後的敗者組積分文字，直接注入 LLM 上下文。
    """
    try:
        rows = _fetch_csv_rows("GOOGLE_SHEET_CSV_LOSER_STANDINGS")
    except RuntimeError as e:
        return f"[敗者組積分查詢失敗] {e}"

    if group_name:
        normalized_query = group_name.strip().lower()
        rows = [r for r in rows if len(r) > _LOS_GROUP and r[_LOS_GROUP].strip().lower() == normalized_query]

    if not rows:
        return f"[敗者組積分查詢結果] 找不到{f'「{group_name}」的' if group_name else ''}敗者組積分資料。"

    lines = [f"[敗者組積分查詢結果]{f'（{group_name}）' if group_name else ''}"]
    current_group = None
    for row in rows:
        if len(row) <= _LOS_STATUS:
            continue
        g = row[_LOS_GROUP].strip()
        if g != current_group:
            lines.append(f"\n【{g}】隊伍 | 平均勝敗局 | 平均全失分 | 狀態")
            current_group = g
        lines.append(
            f"  {row[_LOS_TEAM].strip()} | {row[_LOS_AVG_SETS].strip()} | "
            f"{row[_LOS_AVG_LOST_PTS].strip()} | {row[_LOS_STATUS].strip()}"
        )
    return "\n".join(lines)


# ─── 2. 分組名單查詢 ───────────────────────────────────────────────────────────
#
# 分組名單分頁欄位（A~C）：
#   A：組別（A組/B組/…）  B：隊伍名稱  C：備註（選填）
#
_GRP_GROUP = 0
_GRP_TEAM  = 1
_GRP_NOTE  = 2


def tool_query_groups(group_name: str | None = None) -> str:
    """
    查詢分組名單。

    Parameters
    ----------
    group_name : str | None
        "A組" / "B組" / … / None（全部組別）
    """
    try:
        rows = _fetch_csv_rows("GOOGLE_SHEET_CSV_GROUPS")
    except RuntimeError as e:
        return f"[分組名單查詢失敗] {e}"

    if group_name:
        rows = [r for r in rows if len(r) > _GRP_GROUP and r[_GRP_GROUP].strip() == group_name]

    if not rows:
        return f"[分組名單查詢結果] 找不到{f'「{group_name}」的' if group_name else ''}分組資料。"

    lines = [f"[分組名單查詢結果]{f'（{group_name}）' if group_name else ''}"]
    current_group = None
    for row in rows:
        if len(row) <= _GRP_TEAM:
            continue
        g = row[_GRP_GROUP].strip()
        if g != current_group:
            lines.append(f"\n【{g}】")
            current_group = g
        note = f"（{row[_GRP_NOTE].strip()}）" if len(row) > _GRP_NOTE and row[_GRP_NOTE].strip() else ""
        lines.append(f"  ・{row[_GRP_TEAM].strip()}{note}")
    return "\n".join(lines)


# ─── 3. 積分排名查詢 ───────────────────────────────────────────────────────────
#
# 積分表分頁欄位（A~G）：
#   A：組別  B：排名  C：隊伍  D：勝場  E：負場  F：積分  G：局勝率
#
_STD_GROUP    = 0
_STD_RANK     = 2
_STD_TEAM     = 1
_STD_WIN      = 6
_STD_LOSE     = 7
_STD_PTS      = 4
_STD_SET_RATE = 5


def tool_query_standings(group_name: str | None = None) -> str:
    """
    查詢積分排名。

    Parameters
    ----------
    group_name : str | None
        "A組" / None（全部）
    """
    try:
        rows = _fetch_csv_rows("GOOGLE_SHEET_CSV_STANDINGS")
    except RuntimeError as e:
        return f"[積分表查詢失敗] {e}"

    if group_name:
        rows = [r for r in rows if len(r) > _STD_GROUP and r[_STD_GROUP].strip() == group_name]

    if not rows:
        return f"[積分表查詢結果] 找不到{f'「{group_name}」的' if group_name else ''}積分資料。"

    lines = [f"[積分表查詢結果]{f'（{group_name}）' if group_name else ''}"]
    current_group = None
    for row in rows:
        if len(row) <= _STD_PTS:
            continue
        g = row[_STD_GROUP].strip()
        if g != current_group:
            lines.append(f"\n【{g}】排名 | 隊伍 | 勝/負 | 積分 | 勝敗局商")
            current_group = g
        set_rate = row[_STD_SET_RATE].strip() if len(row) > _STD_SET_RATE else "-"
        lines.append(
            f"  第{row[_STD_RANK].strip()}名 | {row[_STD_TEAM].strip()} | "
            f"{row[_STD_WIN].strip()}勝{row[_STD_LOSE].strip()}負 | "
            f"{row[_STD_PTS].strip()}分 | {set_rate}"
        )
    return "\n".join(lines)


# ─── 4. 晉級/淘汰狀況查詢 ─────────────────────────────────────────────────────
#
# 晉級淘汰分頁欄位（A~C）：
#   A：組別  B：隊伍  C：狀態（晉級/淘汰）
#
_ELM_GROUP  = 0
_ELM_TEAM   = 1
_ELM_STATUS = 2


def tool_query_elimination(group_name: str | None = None) -> str:
    """
    查詢晉級與淘汰結果。

    Parameters
    ----------
    group_name : str | None
        "A組" / None（全部）
    """
    try:
        rows = _fetch_csv_rows("GOOGLE_SHEET_CSV_ELIMINATION")
    except RuntimeError as e:
        return f"[晉級淘汰查詢失敗] {e}"

    if group_name:
        rows = [r for r in rows if len(r) > _ELM_GROUP and r[_ELM_GROUP].strip() == group_name]

    if not rows:
        return f"[晉級淘汰查詢結果] 找不到{f'「{group_name}」的' if group_name else ''}晉淘資料（可能尚未更新）。"

    advanced = [r for r in rows if len(r) > _ELM_STATUS and r[_ELM_STATUS].strip() == "晉級"]
    eliminated = [r for r in rows if len(r) > _ELM_STATUS and r[_ELM_STATUS].strip() == "淘汰"]

    lines = [f"[晉級淘汰查詢結果]{f'（{group_name}）' if group_name else ''}"]
    if advanced:
        lines.append("\n✅ 晉級隊伍：")
        for r in advanced:
            lines.append(f"  ・{r[_ELM_GROUP].strip()} {r[_ELM_TEAM].strip()}")
    if eliminated:
        lines.append("\n❌ 淘汰隊伍：")
        for r in eliminated:
            lines.append(f"  ・{r[_ELM_GROUP].strip()} {r[_ELM_TEAM].strip()}")
    return "\n".join(lines)

