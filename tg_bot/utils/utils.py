import datetime
import logging
from typing import TypeAlias, Optional, Tuple
from domain.domain import Task, Event

# Task aliases
TaskName: TypeAlias = str
Duration: TypeAlias = int
Importance: TypeAlias = int
Complexity: TypeAlias = int
StartTime: TypeAlias = datetime.time
Date: TypeAlias = datetime.date

# Base info aliases
EndTime: TypeAlias = datetime.time

# Event alisases
EventName: TypeAlias = str
EventRepeatArguments: TypeAlias = str

EventTuple: TypeAlias = Tuple[
    EventName, StartTime, Duration, Optional[EventRepeatArguments]
]

# Mark History
IsDone: TypeAlias = bool


def parse_int(int_str: str) -> Optional[int]:
    if not isinstance(int_str, str):
        logging.warning("Got unexpected type")
        return None

    if not int_str.isdigit():
        logging.warning("Got unexpected string")
        return None

    return int(int_str)


def parse_date(date_str: str) -> Optional[Date]:
    if not isinstance(date_str, str):
        logging.warning("Got unexpected type")
        return None

    date_split = date_str.split("/")
    if len(date_split) != 3:
        logging.warning("Cannot parse to date")
        return None

    day, month, year = map(parse_int, date_split)

    if None in [day, month, year]:
        return None

    try:
        date = datetime.date(year, month, day)
        return date
    except ValueError:
        logging.warning("Wrong format of date")

    return None


def parse_time(time_str: str) -> Optional[datetime.time]:
    time_str = time_str.lower()

    if len(time_str.split(":")) != 2:
        logging.warning("Time is in wrong format")
        return None

    is_in_12_format = time_str.endswith("am") or time_str.endswith("pm")
    endfix = time_str[-2:]

    if is_in_12_format:
        time_str = time_str[:-2]

    hours, minutes = map(parse_int, time_str.split(":"))

    if None in [hours, minutes]:
        return None

    if ((hours > 11 and is_in_12_format) or hours > 23) or minutes > 59:
        logging.warning("Hours or minutes are in wrong format")
        return None

    if is_in_12_format and endfix == "pm":
        hours += 12

    return datetime.time(hour=hours, minute=minutes)


def start_end_of_day_time(time_str: str) -> Optional[Tuple[StartTime, EndTime]]:
    if not isinstance(time_str, str):
        logging.warning("Got non-string type in parse time function")
        return None

    time_str = time_str.lower()

    if len(time_str.split("-")) != 2:
        logging.warning("Got wrong format of string")
        return None

    start_time, end_time = map(parse_time, time_str.split("-"))

    if None in [start_time, end_time]:
        return None

    return start_time, end_time


def parse_task(
    task_str: str,
) -> Optional[
    Tuple[
        TaskName,
        Duration,
        Importance,
        Complexity,
        Optional[StartTime],
        Optional[Date],
    ]
]:
    if not isinstance(task_str, str):
        logging.warning("Got non-string type in parse task function")
        return None

    task_split = task_str.split("-")
    if not (4 <= len(task_split) <= 6):
        logging.warning("Got unexpected number of arguments")
        return None

    task_name, duration_str, importance_str, complexity_str = task_split[:4]
    start_time, date = None, None

    if len(task_split) == 6:
        start_time_str, date_str = task_split[4:]

        start_time, date = parse_time(start_time_str), parse_date(date_str)

        if None in [start_time, date]:
            logging.warning("Wrong format of time or date")
            return None

    elif len(task_split) == 5:
        temp_str = task_split[4]

        temp_time = parse_time(temp_str)
        if temp_time is None:
            temp_date = parse_date(temp_str)

            if temp_date is None:
                return None

            date = temp_date

        else:
            start_time = temp_time

    duration, importance, complexity = map(
        parse_int, [duration_str, importance_str, complexity_str]
    )
    if None in [duration, importance, complexity]:
        return None

    return task_name, duration, importance, complexity, start_time, date


def get_task(data: dict) -> Optional[Task]:
    init_dict = {}
    params = ["task_name", "task_duration", "task_importance", "task_complexity"]
    optional_params = ["task_start_time", "task_date"]

    for param in params:
        if param not in data.keys():
            logging.warning("Cannot parse task")
            return None

    init_dict = {
        "task_name": data.get("task_name"),
        "duration": data.get("task_duration"),
        "importance": data.get("task_importance"),
        "complexity": data.get("task_complexity"),
        "start_time": data.get("task_start_time"),
        "date": data.get("task_date"),
    }

    return Task(**init_dict)


def parse_event(event_str: str) -> Optional[Event]:
    if not isinstance(event_str, str):
        logging.warning("Got unexpected type")
        return None

    event_split = event_str.split("-")

    if not (3 <= len(event_split) <= 4):
        logging.warning("Got unexpected number of arguments")
        return None

    event_name, start_time_str, duration_str = event_split[:3]
    repeat_argument = None

    if len(event_split) == 4:
        repeat_argument = event_split[3]

    start_time, duration = parse_time(start_time_str), parse_int(duration_str)

    if None in [start_time, duration]:
        logging.warning("Wrong format of time or duration")
        return None

    return Event(
        event_name=event_name,
        start_time=start_time,
        duration=duration,
        repeat_arguments=repeat_argument,
    )


def parse_bool(bool_str: str) -> Optional[bool]:
    if not isinstance(bool_str, str):
        logging.warning("Got unexpected type")
        return None

    bool_str = bool_str.strip().lower()

    if bool_str == "true":
        return True

    if bool_str == "false":
        return False

    return None


def parse_marking_history(
    mark_history_str: str,
) -> Optional[Tuple[StartTime, EndTime, IsDone]]:
    if not isinstance(mark_history_str, str):
        logging.warning("Got unexpected type")
        return None

    mark_history_split = mark_history_str.split("-")

    if len(mark_history_split) != 3:
        logging.warning("Got unexpected number of arguments")
        return None

    start_time_str, end_time_str, is_done_str = mark_history_split

    start_time, end_time, is_done = (
        parse_time(start_time_str),
        parse_time(end_time_str),
        parse_bool(is_done_str),
    )

    if None in [start_time, end_time, is_done]:
        logging.warning("Wrong format of mark history")
        return None

    return start_time, end_time, is_done
