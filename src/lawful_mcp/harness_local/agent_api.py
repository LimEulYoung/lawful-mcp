"""The date line the Lawful Agent API appends to its system prompt (server/agent_engine.py at the commit named in
harness/README.md). The experiment's system prompt is not part of the harness: both tool-using conditions share
eval/prompts/agent_system.txt.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

# agent_engine._WEEKDAY_KO, _DAY_PARTS
_WEEKDAY_KO = ("월", "화", "수", "목", "금", "토", "일")
_DAY_PARTS = ((5, "새벽"), (11, "오전"), (17, "오후"), (23, "밤"))


def today_hint(now: datetime | None = None) -> str:
    """agent_engine._today_hint: today's date (KST) and the part of the day, in brackets on one line."""
    now = now or datetime.now(ZoneInfo("Asia/Seoul"))
    part = next(label for end, label in _DAY_PARTS if now.hour <= end)
    return f"\n\n[오늘: {now:%Y년 %m월 %d일} ({_WEEKDAY_KO[now.weekday()]}) {part}]"
