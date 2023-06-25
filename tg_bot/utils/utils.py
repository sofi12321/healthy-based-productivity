import datetime
import logging
from typing import TypeAlias
from domain.domain import Task

# Task aliases
TaskName: TypeAlias = str
Duration: TypeAlias = int
Importance: TypeAlias = int
Complexity: TypeAlias = int
StartTime: TypeAlias = datetime.time
Date: TypeAlias = datetime.date

# Base info aliases
EndTime: TypeAlias = datetime.time


def parse_int(int_str: str) -> int or None:
    if not isinstance(int_str, str):
        logging.warning("Got unexpected type")
        return None

    if not int_str.isdigit():
        logging.warning("Got unexpected string")
        return None

    return int(int_str)


def parse_date(date_str: str) -> Date or None:
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


def parse_time(time_str: str) -> datetime.time or None:
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


def start_end_of_day_time(time_str: str) -> (StartTime, EndTime) or None:
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
) -> (
    TaskName,
    Duration,
    Importance,
    Complexity,
    StartTime or None,
    Date or None,
) or None:
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


def get_task(data: dict) -> Task or None:
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
