import datetime
import logging


def get_time(time_str: str) -> datetime.time or None:
    time_str = time_str.lower()

    if len(time_str.split(":")) != 2:
        logging.warning("Time is in wrong format")
        return None

    is_in_12_format = time_str.endswith("am") or time_str.endswith("pm")
    endfix = time_str[-2:]

    if is_in_12_format:
        time_str = time_str[:-2]

    hours_str, minutes_str = time_str.split(":")

    if not hours_str.isdigit() or not minutes_str.isdigit():
        logging.warning("Hours or minutes are in wrong format")
        return None

    hours, minutes = int(hours_str), int(minutes_str)

    if ((hours > 11 and is_in_12_format) or hours > 23) or minutes > 59:
        logging.warning("Hours or minutes are in wrong format")
        return None

    if is_in_12_format and endfix == "pm":
        hours += 12

    return datetime.time(hour=hours, minute=minutes)


def parse_time(time_str: str) -> (datetime.time, datetime.time) or None:
    if not isinstance(time_str, str):
        logging.warning("Got non-string type in parse time function")
        return None

    time_str = time_str.lower()

    if len(time_str.split("-")) != 2:
        logging.warning("Got wrong format of string")
        return None

    start_time, end_time = map(get_time, time_str.split("-"))

    if None in [start_time, end_time]:
        return None

    return start_time, end_time
