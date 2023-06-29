import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    telegam_id: int
    task_name: str
    duration: int
    importance: int
    complexity: int
    start_time: Optional[datetime.time] = None
    date: datetime.date = datetime.date.today()
    is_done: bool = False
    real_start: Optional[datetime.time] = None
    real_duration: Optional[int] = None
    real_date: Optional[datetime.date] = None
    task_id: Optional[int] = None


@dataclass
class BasicUserInfo:
    telegram_id: int
    user_name: str
    start_time: datetime.time
    end_time: datetime.time


@dataclass
class Event:
    event_id: Optional[int]
    repeat_number: int = 0
    event_name: str
    start_time: datetime.time
    duration: int
    date: datetime.date = datetime.date.today()
