from aiogram import Bot, Dispatcher
from aiogram.types import Message, CallbackQuery
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import numpy as np
from torch import tensor
import tensorflow as tf

import tg_bot.config as config
from tg_bot.adapters import orm, repositories

from tg_bot.states.states import (
    GetBasicInfo,
    GetTaskInfo,
    GetEventInfo,
    MarkHistory,
    List,
)
from tg_bot.lexicon.lexicon import lexicon, get_lexicon_with_argument
from tg_bot.utils.utils import (
    parse_numpy_arr,
    parse_user,
    parse_task,
    parse_int,
    parse_time,
    parse_date,
    parse_event,
    parse_marking_history,
    parse_repeated_arguments,
    parse_int_in_range,
    check_repeat_each_argument,
    numpy_to_string,
)
from tg_bot.adapters.repositories import get_events_repo, get_tasks_repo, get_users_repo

from tg_bot.keyboards.keyboards import (
    get_active_tasks_keyboard,
    get_start_buttons_keyboard,
)
from tg_bot.domain.domain import Task

from Structure.plan import Planner

import datetime
import os
import logging

API_TOKEN: str = os.getenv("TGTOKEN")

# Number of hours that represents part of time for model to predict data
ALPHA: int = 24
PLANNER: Planner = Planner()

if API_TOKEN is None:
    logging.error(
        "Please enter telegram bot api token to the environment variable 'TGTOKEN'"
    )
    exit(1)

engine = create_engine(config.get_db_uri("tg_bot/local_db/prototype_v1.db"))
get_session = sessionmaker(bind=engine)
repositories.initialize_repositories(get_session())
orm.start_mappers()

# orm.delete_all(engine)
orm.create_all(engine)


bot: Bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp: Dispatcher = Dispatcher(bot, storage=storage)


@dp.message_handler(state="*", commands="start")
async def process_start_command(message: Message, state: FSMContext):
    data = await state.get_data()

    repo = get_users_repo()
    user_id = message.from_id
    user = repo.get_user_by_id(user_id)

    if user is not None:
        await message.reply(
            lexicon["en"]["hello"], reply_markup=await get_start_buttons_keyboard()
        )
        return

    await state.set_state(GetBasicInfo.NameOfUser)
    logging.info("Getting info about start and end of time from user")
    await message.answer(lexicon["en"]["first_hello"])


@dp.message_handler(state=GetBasicInfo.NameOfUser)
async def get_user_name(message: Message, state: FSMContext):
    message_text = message.text

    message_text = message_text.strip()

    if not (1 <= len(message_text) <= 255):
        """Please retry"""
        await message.answer(lexicon["en"]["retry_name"])
        return

    async with state.proxy() as data:
        data["user_name"] = message_text

    await state.set_state(GetBasicInfo.StartOfDay)
    await message.answer(lexicon["en"]["basic_info_start_of_day"])


@dp.message_handler(state=GetBasicInfo.StartOfDay)
async def get_start_day(message: Message, state: FSMContext):
    message_text = message.text
    time = parse_time(message_text)

    if time is None:
        await message.reply(lexicon["en"]["retry_time"])
        return

    async with state.proxy() as data:
        data["start_time"] = time

    await state.set_state(GetBasicInfo.EndOfDay)
    await message.answer(lexicon["en"]["basic_info_end_of_day"])


@dp.message_handler(state=GetBasicInfo.EndOfDay)
async def get_end_day(message: Message, state: FSMContext):
    message_text = message.text
    time = parse_time(message_text)

    if time is None:
        """Please retry"""
        await message.reply(lexicon["en"]["retry_time"])
        return

    async with state.proxy() as data:
        data["end_time"] = time

        history, context = PLANNER.scheduler.get_states()

        history_str, context_str = numpy_to_string(history.numpy()), numpy_to_string(
            context.numpy()
        )

        user = parse_user(
            user_id=message.from_id,
            user_name=data.get("user_name"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            history=history_str,
            context=context_str,
        )

    if user is None:
        logging.error(get_lexicon_with_argument("error", data.as_dict()))
        await message.answer(
            lexicon["en"]["retry_trouble"],
            reply_markup=await get_start_buttons_keyboard(),
        )
        await state.finish()
        return

    try:
        repo = get_users_repo()
        repo.add_user(user)
    except Exception as e:
        logging.error(get_lexicon_with_argument("error", e.args))
        await message.answer(
            lexicon["en"]["retry_trouble"],
            reply_markup=await get_start_buttons_keyboard(),
        )
        await state.finish()
        return

    await message.answer(
        lexicon["en"]["write_success"], reply_markup=await get_start_buttons_keyboard()
    )
    await state.finish()


@dp.message_handler(state="*", commands=["add_task"])
async def add_task(message: Message, state: FSMContext):
    repo = get_users_repo()
    user = repo.get_user_by_id(message.from_id)

    if user is None:
        """Please enter '/start'"""
        await message.answer(lexicon["en"]["user_not_found"])
        return

    await state.set_state(GetTaskInfo.TaskNameState)
    await message.answer(lexicon["en"]["add_task"])


@dp.message_handler(state=GetTaskInfo.TaskNameState)
async def get_task_name(message: Message, state: FSMContext):
    """
    Used to parse beggining with gaining task name
    """
    message_text = message.text
    message_text = message_text.strip()

    if not (1 <= len(message_text) <= 255):
        """Please retry"""
        await message.answer(lexicon["en"]["retry_name"])

    async with state.proxy() as data:
        data["task_name"] = message_text

    await state.set_state(GetTaskInfo.TaskDurationState)
    await message.answer(lexicon["en"]["get_duration_task"])


@dp.message_handler(state=GetTaskInfo.TaskDurationState)
async def get_task_duration(message: Message, state: FSMContext):
    message_text = message.text
    result = parse_int_in_range(message_text, start_range=1, end_range=24 * 60)

    if result is None:
        """Please retry"""
        await message.answer(lexicon["en"]["retry_int"])
        return

    async with state.proxy() as data:
        data["task_duration"] = result

    await state.set_state(GetTaskInfo.TaskImportanceState)
    await message.answer(lexicon["en"]["get_importance_task"])


@dp.message_handler(state=GetTaskInfo.TaskImportanceState)
async def get_task_duration(message: Message, state: FSMContext):
    message_text = message.text
    result = parse_int_in_range(message_text, start_range=0, end_range=3)

    if result is None:
        """Please retry"""
        await message.answer(lexicon["en"]["retry_int"])
        return

    async with state.proxy() as data:
        data["task_importance"] = result

    await state.set_state(GetTaskInfo.TaskStartTimeState)
    await message.answer(lexicon["en"]["get_start_time_task"])


@dp.message_handler(state=GetTaskInfo.TaskStartTimeState)
async def get_task_start_time(message: Message, state: FSMContext):
    message_text = message.text

    if message_text == "no":
        await state.set_state(GetTaskInfo.TaskDateState)
        await message.answer(lexicon["en"]["get_date_task"])
        return

    result = parse_time(message_text)

    if result is None:
        """Please retry"""
        await message.answer(lexicon["en"]["retry_optional_time"])
        return

    async with state.proxy() as data:
        data["task_start_time"] = result

    await state.set_state(GetTaskInfo.TaskDateState)
    await message.answer(lexicon["en"]["get_date_task"])


@dp.message_handler(state=GetTaskInfo.TaskDateState)
async def get_task_date(message: Message, state: FSMContext):
    message_text = message.text

    if message_text != "no":
        result = parse_date(message_text)

        if result is None:
            """Please retry"""
            await message.answer(lexicon["en"]["retry_optional_date"])
            return

    repo = get_tasks_repo()
    async with state.proxy() as data:
        task = parse_task(
            user_id=message.from_id,
            task_name=data.get("task_name"),
            duration=data.get("task_duration"),
            importance=data.get("task_importance"),
            start_time=data.get("task_start_time"),
            date=data.get("task_date"),
        )

    if task is None:
        # Error
        logging.error(get_lexicon_with_argument("error", data.as_dict()))
        await message.answer(
            lexicon["en"]["retry_trouble"],
            reply_markup=await get_start_buttons_keyboard(),
        )
        await state.finish()
        return

    try:
        repo.add_task(task)
    except Exception as e:
        # Error
        logging.error(get_lexicon_with_argument("error", e.args))
        await message.answer(
            lexicon["en"]["retry_trouble"],
            reply_markup=await get_start_buttons_keyboard(),
        )
        await state.finish()
        return

    await state.finish()
    await message.answer(
        lexicon["en"]["write_success"], reply_markup=await get_start_buttons_keyboard()
    )


@dp.message_handler(state="*", commands=["add_event"])
async def add_event(message: Message, state: FSMContext):
    message_text = message.text

    repo = get_users_repo()
    user = repo.get_user_by_id(message.from_id)

    if user is None:
        """Please enter '/start'"""
        await message.answer(lexicon["en"]["user_not_found"])
        return

    await state.set_state(GetEventInfo.EventNameState)
    await message.answer(lexicon["en"]["add_event"])


@dp.message_handler(state=GetEventInfo.EventNameState)
async def get_event_name(message: Message, state: FSMContext):
    message_text = message.text
    message_text = message_text.strip()

    if not (1 <= len(message_text) <= 255):
        """Please retry"""
        await message.answer(lexicon["en"]["retry_name"])
        return

    async with state.proxy() as data:
        data["event_name"] = message_text

    await state.set_state(GetEventInfo.EventStartTimeState)
    await message.answer(lexicon["en"]["event_start_time"])


@dp.message_handler(state=GetEventInfo.EventStartTimeState)
async def get_event_start_time(message: Message, state: FSMContext):
    message_text = message.text

    result = parse_time(message_text)

    if result is None:
        """Please retry"""
        await message.answer(lexicon["en"]["retry_time"])
        return

    async with state.proxy() as data:
        data["event_start_time"] = result

    await state.set_state(state=GetEventInfo.EventDurationState)
    await message.answer(lexicon["en"]["event_duration"])


@dp.message_handler(state=GetEventInfo.EventDurationState)
async def get_event_duration(message: Message, state: FSMContext):
    message_text = message.text
    result = parse_int_in_range(message_text, start_range=1, end_range=24 * 60)

    if result is None:
        """Please retry"""
        await message.answer(lexicon["en"]["retry_int"])
        return

    async with state.proxy() as data:
        data["event_duration"] = result

    await state.set_state(GetEventInfo.EventDateState)
    await message.answer(lexicon["en"]["event_date"])


@dp.message_handler(state=GetEventInfo.EventDateState)
async def get_event_date(message: Message, state: FSMContext):
    message_text = message.text
    message_text = message_text.strip()

    if message_text.lower() != "no":
        result = parse_date(message_text)

        if result is None:
            """Please retry"""
            await message.answer(lexicon["en"]["retry_optional_date"])
            return

        async with state.proxy() as data:
            data["event_date"] = result

    await state.set_state(state=GetEventInfo.EventRepeatIsNeeded)
    await message.answer(lexicon["en"]["event_want_repeat"])


@dp.message_handler(state=GetEventInfo.EventRepeatIsNeeded)
async def event_want_repeat(message: Message, state: FSMContext):
    message_text = message.text
    message_text = message_text.strip().lower()

    result = None

    if message_text == "yes":
        result = True

    if message_text == "no":
        result = False

    if result is None:
        """Please retry"""

        await message.answer(lexicon["en"]["retry_yesno"])
        return

    if result:
        await message.answer(lexicon["en"]["event_repeat_each_number"])
        await state.set_state(GetEventInfo.EventRepeatEachNumberState)
        return

    async with state.proxy() as data:
        dates = data.get("event_date")

        if dates is not None:
            dates = [dates]

        event = parse_event(
            user_id=message.from_id,
            event_name=data.get("event_name"),
            start_time=data.get("event_start_time"),
            duration=data.get("event_duration"),
            dates=dates,
        )

        if event is None:
            logging.error(
                get_lexicon_with_argument(
                    "error", f"Could not parse event: {data.as_dict()}"
                )
            )
            await message.answer(
                lexicon["en"]["retry_trouble"],
                reply_markup=await get_start_buttons_keyboard(),
            )
            await state.finish()
            return

    repo = get_events_repo()
    try:
        repo.add_event(event)
    except Exception as e:
        logging.error(
            get_lexicon_with_argument("error", f"Could not save to db: {e.args}")
        )
        await message.answer(
            lexicon["en"]["retry_trouble"],
            reply_markup=await get_start_buttons_keyboard(),
        )
        await state.finish()
        return

    await message.answer(
        lexicon["write_success"], reply_markup=await get_start_buttons_keyboard()
    )
    await state.finish()


@dp.message_handler(state=GetEventInfo.EventRepeatEachNumberState)
async def get_event_repeat_each_number(message: Message, state: FSMContext):
    message_text = message.text
    message_text = message_text.strip()

    if message_text.lower() == "no":
        """TODO: Get event, write event to db"""
        await message.answer(
            lexicon["en"]["write_success"],
            reply_markup=await get_start_buttons_keyboard(),
        )
        await state.finish()
        return

    result = parse_int_in_range(message_text, start_range=1, end_range=15)

    if result is None:
        """Please retry"""
        await message.answer(lexicon["en"]["retry_int"])
        return

    async with state.proxy() as data:
        data["event_each_number"] = result

    await state.set_state(state=GetEventInfo.EventRepeatEachArgumentState)
    await message.answer(lexicon["en"]["event_repeat_each_argument"])


@dp.message_handler(state=GetEventInfo.EventRepeatEachArgumentState)
async def get_event_repeat_each_arguments(message: Message, state: FSMContext):
    message_text = message.text
    message_text = message_text.strip().lower()

    if not check_repeat_each_argument(message_text):
        """Please retry, TODO. What todo?"""
        await message.answer(lexicon["en"]["event_repeat_each_argument"])
        return

    async with state.proxy() as data:
        data["event_each_argument"] = message_text

    await state.set_state(GetEventInfo.EventRepeatNumberOfRepetitions)
    await message.answer(lexicon["en"]["event_repeat_number_of_repetitions"])


@dp.message_handler(state=GetEventInfo.EventRepeatNumberOfRepetitions)
async def get_event_repeat_each_number(message: Message, state: FSMContext):
    message_text = message.text

    result = parse_int_in_range(message_text, start_range=1, end_range=15)

    if result is None:
        """Please retry"""
        await message.answer(lexicon["en"]["retry_int"])
        return

    async with state.proxy() as data:
        dates = parse_repeated_arguments(
            data.get("event_date"),
            tuple(
                [data.get("event_each_number"), data.get("event_each_argument"), result]
            ),
        )

        if dates is None:
            logging.error(
                get_lexicon_with_argument(
                    "error", f"Could not parse dates: \n\t{data.as_dict()}\n\t{result}"
                )
            )
            await message.answer(
                lexicon["en"]["retry_trouble"],
                reply_markup=await get_start_buttons_keyboard(),
            )
            await state.finish()
            return

        events = parse_event(
            user_id=message.from_id,
            event_name=data.get("event_name"),
            start_time=data.get("event_start_time"),
            duration=data.get("event_duration"),
            dates=dates,
        )

        if events is None:
            logging.error(
                get_lexicon_with_argument(
                    "error", f"Could not parse events: \n\t{dates},\n\t{data.as_dict()}"
                )
            )
            await message.answer(
                lexicon["en"]["retry_trouble"],
                reply_markup=await get_start_buttons_keyboard(),
            )
            await state.finish()
            return
    try:
        repo = get_events_repo()
        repo.add_many_events(events)
    except Exception as e:
        logging.error(get_lexicon_with_argument("error", e.args))
        await message.answer(
            lexicon["en"]["retry_trouble"],
            reply_markup=await get_start_buttons_keyboard(),
        )
        await state.finish()
        return

    await state.finish()
    await message.answer(
        lexicon["en"]["write_success"], reply_markup=await get_start_buttons_keyboard()
    )


@dp.message_handler(commands=["mark_history"])
async def mark_history(message: Message, state: FSMContext):
    message_text = message.text

    repo = get_users_repo()
    user = repo.get_user_by_id(message.from_id)

    if user is None:
        """Please enter '/start'"""
        await message.answer(lexicon["en"]["user_not_found"])
        return

    repo = get_tasks_repo()

    async with state.proxy() as data:
        keyboard = await get_active_tasks_keyboard(
            repo.get_active_tasks(user.telegram_id)
        )
        await message.answer(lexicon["en"]["mark_history"], reply_markup=keyboard)

    await state.set_state(MarkHistory.ChooseTask)


@dp.callback_query_handler(
    lambda c: c.data.startwith("task_"), state=MarkHistory.ChooseTask
)
async def mark_history_to_task(callback_query: CallbackQuery, state: FSMContext):
    message_text: str = callback_query.message.text
    task_id_str = callback_query.data.removeprefix("task_")
    task_id = parse_int(task_id_str)

    user_id = callback_query.from_user.id

    repo = get_tasks_repo()

    async with state.proxy() as data:
        task = repo.get_task_by_id(task_id=task_id, user_id=user_id)
        if task is None:
            await callback_query.answer("No tasks")
            return
        else:
            await callback_query.message.answer(
                f"""Task:
                Task name: {task.task_name}\t
                Task duration: {task.duration}\t
                Task importance: {task.importance}\t
                Task start time: {task.start_time or "No time"}\t
                Task date: {task.date or "No date"}\n"""
            )
            await callback_query.message.answer(
                lexicon["en"]["marking_history_is_done"]
            )

    await state.set_state(MarkHistory.MarkingHistoryIsDone)

    async with state.proxy() as data:
        data["marking_task"] = task


@dp.message_handler(state=MarkHistory.MarkingHistoryStartTime)
async def marking_history_start_time(message: Message, state: FSMContext):
    message_text = message.text

    result = parse_time(message_text)

    if result is None:
        await message.answer(lexicon["en"]["retry_time"])
        return

    async with state.proxy() as data:
        data["start_time"] = result

    await state.set_state(MarkHistory.MarkingHistoryDuration)
    await message.answer(lexicon["en"]["marking_history_duration"])


@dp.message_handler(state=MarkHistory.MarkingHistoryDuration)
async def marking_history_duration(message: Message, state: FSMContext):
    message_text = message.text

    result = parse_int_in_range(message_text, start_range=1, end_range=24 * 60)

    if result is None:
        await message.answer(lexicon["en"]["retry_int"])
        return

    repo = get_tasks_repo()

    async with state.proxy() as data:
        data["duration"] = result

        task: Task = data.get("marking_task")

        task.real_start = data.get("start_time")
        task.real_date = data.get("date", datetime.date.today())
        task.real_duration = data.get("duration")
        task.is_done = data.get("is_done", True)
        repo.add_task(task)

    await state.finish()
    await message.answer(lexicon["en"]["write_success"])


@dp.message_handler(state=MarkHistory.MarkingHistoryIsDone)
async def marking_history_is_done(message: Message, state: FSMContext):
    message_text = message.text
    message_text = message_text.strip().lower()

    result = None

    if message_text == "yes":
        result = True

    if message_text == "no":
        result = False

    if result is None:
        """Please retry"""
        await message.answer(lexicon["en"]["retry_yesno"])
        return

    repo = get_tasks_repo()

    async with state.proxy() as data:
        data["is_done"] = result

        task: Task = data.get("marking_task")

        if data["is_done"] is False:
            task: Task = data.get("marking_task")

    await message.answer(
        lexicon["en"]["write_success"], reply_markup=await get_start_buttons_keyboard()
    )
    await state.finish()


@dp.message_handler(commands=["plan"], state="*")
async def plan_new_schedule(message: Message, state: FSMContext):
    """
    Gathering unscheduled tasks and events from user and
    sends it to model and shows new generated schedule
    by model
    """

    message_text = message.text

    await message.answer(lexicon["en"]["plan"])

    task_repo = get_tasks_repo()
    event_repo = get_events_repo()
    user_repo = get_users_repo()

    """Gathering unscheduled tasks and events"""

    try:
        user = user_repo.get_user_by_id(message.from_id)

        if user is None:
            raise ValueError("Could not find user that made message")

    except Exception as e:
        logging.error(get_lexicon_with_argument("error", e.args))
        return

    print(user.history, user.context)

    user_history, user_context = parse_numpy_arr(user.history), parse_numpy_arr(
        user.context
    )

    current_time = datetime.datetime.now()

    tasks = task_repo.get_task_in_range(
        message.from_id, ALPHA, current_time.date(), current_time.time()
    )
    events = event_repo.get_events_in_range(
        message.from_id, ALPHA, current_time.date(), current_time.time()
    )

    tasks = list(filter(lambda task: task.predicted_date is None, tasks))
    events = list(filter(lambda event: event.predicted_date is None, events))

    """Sending to model"""

    new_schedule, user_h, user_c = PLANNER.get_model_schedule(
        tasks, events, user_history, user_context
    )

    """Gathering and saving generated information"""

    for new_task_info in new_schedule:
        try:
            task = task_repo.get_task_by_id(
                task_id=new_task_info["task_id"], user_id=message.from_id
            )

            if task is None:
                logging.warning("Got unexpected output from db")
                continue

            task.predicted_start = new_task_info["predicted_start_time"]
            task.predicted_offset = new_task_info["predicted_offset"]
            task.predicted_date = new_task_info["predicted_date"]
            task.predicted_duration = new_task_info["predicted_duration"]

            task_repo.add_task(task)
        except Exception as e:
            logging.error(get_lexicon_with_argument("error", e.args))
            await message.answer(lexicon["en"]["retry_trouble"])
            continue

    """Replying to user"""

    tasks = task_repo.get_task_in_range(
        message.from_id, ALPHA, current_time.date(), current_time.time()
    )
    events = event_repo.get_events_in_range(
        message.from_id, ALPHA, current_time.date(), current_time.time()
    )

    await message.answer(PLANNER.print_schedule(tasks, events))


@dp.message_handler(state="*", commands=["list"])
def show_tasks(message: Message, state: FSMContext):
    message_text = message.text

    user = get_users_repo().get_user_by_id(message.from_id)

    if user is None:
        """Please '/start'"""
        await message.answer(lexicon["en"]["user_not_found"])
        return

    await message.answer(lexicon["en"]["list_date"])
    await state.set_state(List.ChooseDate)


@dp.message_handler(state=List.ChooseDate)
def choose_date_to_list(message: Message, state: FSMContext):
    message_text = message.text

    result = parse_date(message_text)

    if message_text.strip().lower() == "today":
        result = datetime.date.today()

    if message_text.strip().lower() == "tomorrow":
        result = datetime.date.today() + datetime.timedelta(days=1)

    if result is None:
        """Please retry"""
        await message.answer(lexicon["en"]["retry_date"])
        return

    async with state.proxy() as data:
        data["list_date"] = result

    await message.answer(lexicon["en"]["list_task_event"])
    await state.set_state(List.ListingTasksEvents)


# TODO: list
