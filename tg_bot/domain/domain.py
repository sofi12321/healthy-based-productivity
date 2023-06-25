import datetime
from dataclasses import dataclass
from typing import Optional


"""
class Task:
    def __init__(self, task_name, duration, importance, start_time=None, date=None):
        self.task_name: str = task_name
        self.duration: int = duration
        self.importance: int = importance
        self.start_time: datetime.time = start_time
        self.date: datetime.date = date
"""


@dataclass
class Task:
    task_name: str
    duration: int
    importance: int
    complexity: int
    start_time: Optional[datetime.time] = None
    date: Optional[datetime.date] = None


@dataclass
class BasicUserInfo:
    start_time: datetime.time
    end_time: datetime.time


@dataclass
class Event:
    event_name: str
    start_time: datetime.time
    duration: int
    repeat_arguments: Optional[str]
