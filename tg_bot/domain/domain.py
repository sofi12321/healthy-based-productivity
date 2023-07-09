import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    telegram_id: int
    task_name: str
    duration: int
    importance: int
    start_time: Optional[datetime.time] = None
    date: datetime.date = datetime.date.today()
    is_done: bool = False
    real_start: Optional[datetime.time] = None
    real_duration: Optional[int] = None
    real_date: Optional[datetime.date] = None
    task_id: Optional[int] = None
    predicted_start: Optional[datetime.time] = None
    predicted_offset: Optional[int] = None
    predicted_duration: Optional[int] = None
    predicted_date: Optional[datetime.date] = None


@dataclass
class BasicUserInfo:
    telegram_id: int
    user_name: str
    start_time: datetime.time
    end_time: datetime.time
    history: str
    context: str


@dataclass
class Event:
    telegram_id: int
    event_name: str
    start_time: datetime.time
    duration: int
    date: datetime.date = datetime.date.today()
    repeat_number: int = 0
    event_id: Optional[int] = None
    was_scheduled: bool = False