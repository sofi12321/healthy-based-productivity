import datetime
import logging
import numpy as np
from typing import TypeAlias, Optional, Tuple, List
from tg_bot.domain.domain import Task, Event, BasicUserInfo

from torch import tensor, float32, Tensor

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


def numpy_to_string(numpy_arr: np.float32) -> str:
    numpy_arr = numpy_arr.reshape(1, -1)[0]
    new_arr = []
    for elem in numpy_arr:
        new_arr.append(str(elem))

    return ' '.join(new_arr)


def parse_numpy_arr(numpy_arr_str: str) -> Optional[Tensor]:
    if not isinstance(numpy_arr_str, str):
        logging.warning("Got unexpected type")
        return None

    return tensor(np.array([np.fromstring(numpy_arr_str, dtype=float, sep=' ')]), dtype=float32)


def day_of_week(day_str: str) -> Optional[int]:
    if not isinstance(day_str, str):
        logging.warning("Got unexpected type")
        return None

    day_str = day_str.strip().lower()

    days_of_week = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]

    return days_of_week.index(day_str) if day_str in days_of_week else None


def check_repeat_each_argument(repeat_argument: str) -> Optional[bool]:
    if not isinstance(repeat_argument, str):
        logging.warning("Got unexpected type")
        return None

    repeat_argument = repeat_argument.strip().lower()

    return day_of_week(repeat_argument) is not None or repeat_argument in ['day', 'week', 'month']


def parse_int(int_str: str) -> Optional[int]:
    if not isinstance(int_str, str):
        logging.warning("Got unexpected type")
        return None

    if not int_str.isdigit():
        logging.warning("Got unexpected string")
        return None

    return int(int_str)


def parse_int_in_range(int_str: str, start_range: Optional[int], end_range: Optional[int]) -> Optional[int]:
    """
    Parse int in specific range (range boundaries are included in range)
        start_range is None for no lower boundaries for int
        end_range is None for no upper boundaries for int
    """
    return_int = parse_int(int_str)

    if return_int is None:
        return None

    if not isinstance(start_range, int) and start_range is not None:
        logging.warning("Got unexpected start range")
        return None

    if not isinstance(end_range, int) and end_range is not None:
        logging.warning("Got unexpected end range")
        return None

    if start_range is not None and end_range is not None:
        if start_range > end_range:
            logging.warning("start range is more than end_range")
            return None

    if start_range is not None:
        if return_int < start_range:
            logging.warning("Given number is less than start range")
            return None

    if end_range is not None:
        if return_int > end_range:
            logging.warning("Given number is more than end range")

    return return_int


def parse_complexity(int_str: str) -> Optional[Complexity]:
    return parse_int_in_range(int_str, start_range=0, end_range=3)


def parse_importance(int_str: str) -> Optional[Importance]:
    return parse_int_in_range(int_str, start_range=0, end_range=3)


def get_next_date(
    current_date: Optional[datetime.date], argument: str, each_number: int
) -> Optional[datetime.date]:
    if not (
        (isinstance(current_date, datetime.date) or current_date is None)
        and isinstance(argument, str)
        and isinstance(each_number, int)
    ):
        logging.warning("Got unexpected arguments")
        return None

    argument = argument.strip().lower()

    weekday = day_of_week(argument)

    if argument not in ["day", "month", "week"] and weekday is None:
        logging.warning("Wrong argument for getting next date")
        return None

    if weekday is not None:
        tmp_date = current_date
        if current_date is None:
            current_date = datetime.date.today()

        days_delta = weekday - current_date.weekday()
        if days_delta < 0:
            days_delta += 7

        return_date = current_date + datetime.timedelta(days=days_delta)

        if return_date == current_date and tmp_date is not None:
            return current_date + datetime.timedelta(days=7 * each_number)

        return return_date

    if argument == "day":
        return current_date + datetime.timedelta(days=each_number)

    if argument == "week":
        return current_date + datetime.timedelta(days=7 * each_number)

    if argument == "month":
        new_month = current_date.month + 1
        new_year = current_date.year
        new_day = current_date.day

        if new_month > 12:
            new_month = 1
            new_year += 1

        try:
            new_date = datetime.date(year=new_year, month=new_month, day=new_day)
            return new_date
        except ValueError:
            return None


def parse_repeated_arguments(
    date: Optional[datetime.date], repeated_arguments: Tuple[int, str, int]
) -> Optional[List[datetime.date]]:
    if (date is not None and not isinstance(date, datetime.date)) or not isinstance(repeated_arguments, tuple
    ):
        logging.warning("Got unexpected agruments")
        return None

    if len(repeated_arguments) != 3:
        logging.warning("Wrong number of arguments for repeated arguments")
        return None

    each_number = repeated_arguments[0]

    if not isinstance(each_number, int) or each_number < 1:
        print(each_number)
        logging.warning("Wrong argument for first argument in repeated events")
        return None

    each_argument = repeated_arguments[1]

    if not isinstance(repeated_arguments[1], str):
        logging.warning("Wrong type of each argument")
        return None

    weekday_number = day_of_week(each_argument)

    if not (weekday_number is not None or each_argument in ["day", "week", "month"]):
        logging.warning("Wrong value of second argument in repeated events")
        return None

    number_of_repetitions = repeated_arguments[2]

    if not isinstance(number_of_repetitions, int):
        logging.warning("Wrong argument for third argument in repeated events")
        return None

    date_list = []

    if date is not None:
        if weekday_number is not None:
            """
            Checking for correspondance of weekday of date
            and weekday of repeat arguments
            """
            if date.weekday() != weekday_number:
                logging.warning("Date is not correspond to repeat arguments")
                return None
        date_list = [date]
    else:
        if weekday_number is not None:
            date_list = [
                get_next_date(datetime.date.today(), each_argument, each_number)
            ]
        else:
            date_list = [datetime.date.today()]

    for i in range(number_of_repetitions - 1):
        next_date = get_next_date(date_list[-1], each_argument, each_number)

        number_of_trying = 1

        while next_date is None and i != number_of_repetitions - 1 - number_of_trying:
            "In case of days in month less than days in date"
            next_date = get_next_date(date_list[-1], each_argument, each_number + number_of_trying)
            number_of_trying += 1

        date_list.append(next_date)

    return date_list


def parse_date(date_str: str) -> Optional[Date]:
    if not isinstance(date_str, str):
        logging.warning("Got unexpected type")
        return None

    date_split = date_str.strip().split("/")
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
    time_str = time_str.strip().lower()

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


"""
def parse_start_end_of_day_time(time_str: str) -> Optional[Tuple[StartTime, EndTime]]:
    if not isinstance(time_str, str):
        logging.warning("Got non-string type in parse time function")
        return None

    time_str = time_str.strip().lower()

    if len(time_str.split("-")) != 2:
        logging.warning("Got wrong format of string")
        return None

    start_time, end_time = map(parse_time, time_str.split("-"))

    if None in [start_time, end_time]:
        return None

    return start_time, end_time
"""


def parse_user(user_id: int, user_name: str, start_time: datetime.time, end_time: datetime.time, history: str, context: str) -> Optional[BasicUserInfo]:
    if not isinstance(user_id, int) or not isinstance(user_name, str) or not isinstance(start_time, datetime.time) or not isinstance(end_time, datetime.time) or not isinstance(history, str) or not isinstance(context, str):
        logging.warning("Got unexpected type")
        return None

    return BasicUserInfo(
        telegram_id=user_id,
        user_name=user_name,
        start_time=start_time,
        end_time=end_time,
        history=history,
        context=context
    )


def parse_task(user_id: int, task_name: str, duration: int, importance: int, start_time: Optional[datetime.time], date: Optional[datetime.date]) -> Optional[Task]:
    if not (isinstance(user_id, int) and isinstance(task_name, str) and isinstance(duration, int) and isinstance(importance, int) and (isinstance(start_time, datetime.time) or start_time is None) and (isinstance(date, datetime.date) or date is None)):
        # logging.debug(user_id, task_name, duration, importance, start_time, date)
        logging.warning("Got unexpected type")
        return None

    task_name = task_name.strip()

    if not (1 <= len(task_name) <= 255):
        logging.warning("Wrong format of task_name")
        return None

    if not (1 <= duration <= 24 * 60):
        logging.warning("Wrong format of duration")
        return None

    if not (0 <= importance <= 3):
        logging.warning("Wrong format of importance")
        return None

    return Task(
        telegram_id=user_id,
        task_name=task_name,
        duration=duration,
        importance=importance,
        start_time=start_time,
        date=date or datetime.date.today(),
    )


"""
def parse_task(task_str: str, telegram_id: int) -> Optional[Task]:
    if not isinstance(task_str, str):
        logging.warning("Got non-string type in parse task function")
        return None

    task_split = task_str.strip().split("-")
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

    duration, importance, complexity = (
        parse_int(duration_str),
        parse_importance(importance_str),
        parse_complexity(complexity_str),
    )

    if None in [duration, importance, complexity]:
        return None

    return Task(
        telegram_id, task_name, duration, importance, complexity, start_time, date
    )
"""


def get_task(data: dict, telegram_id) -> Optional[Task]:
    init_dict = {}
    params = ["task_name", "task_duration", "task_importance", "task_complexity"]
    optional_params = ["task_start_time", "task_date"]

    for param in params:
        if param not in data.keys():
            logging.warning("Cannot parse task")
            return None

    init_dict = {
        "telegram_id": telegram_id,
        "task_name": data.get("task_name"),
        "duration": data.get("task_duration"),
        "importance": data.get("task_importance"),
        "complexity": data.get("task_complexity"),
        "start_time": data.get("task_start_time"),
        "date": data.get("task_date"),
    }

    return Task(**init_dict)


def parse_event(
    user_id: int,
    event_name: str,
    start_time: datetime.time,
    duration: int,
    dates: Optional[datetime.date],
) -> Optional[List[Event]]:
    if (
        
        not isinstance(event_name, str)
        or not isinstance(user_id, int)
        or not isinstance(start_time, datetime.time)
        or not isinstance(duration, int)
        or (dates is not None and not isinstance(dates, list))
    ):
        logging.warning("Wrong type of passed arguments")
        return None

    if duration < 1:
        logging.warning("Wrong value for duration")
        return None

    if dates is None:
        dates = [datetime.date.today()]

    events = [
        Event(
            telegram_id=user_id,
            event_id=None,
            event_name=event_name,
            repeat_number=i,
            start_time=start_time,
            duration=duration,
            date=date_tmp,
        )
        for i, date_tmp in enumerate(dates)
    ]

    return events


"""
def parse_event_str(event_str: str) -> Optional[List[Event]]:
    \"\"\"
    parse event from string
    string should be in the following format:
        [event_name]-[start_time]-[duration]-[date]-[repeat_arguments]
    \"\"\"

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
"""


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

    mark_history_split = mark_history_str.strip().split("-")

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
