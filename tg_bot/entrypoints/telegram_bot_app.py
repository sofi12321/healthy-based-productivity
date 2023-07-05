from aiogram import Bot, Dispatcher
from aiogram.types import Message, CallbackQuery
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config
from adapters import orm, repositories

from states.states import (
    GetBasicInfo,
    GetTaskInfo,
    GetEventInfo,
    MarkHistory,
)
from lexicon.lexicon import lexicon, get_lexicon_with_argument
from utils.utils import (
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
)
from adapters.repositories import get_events_repo, get_tasks_repo, get_users_repo

from keyboards.keyboards import get_active_tasks_keyboard
from domain.domain import Task

import datetime
import os
import logging

API_TOKEN: str = os.getenv("TGTOKEN")

if API_TOKEN is None:
    logging.error(
        "Please enter telegram bot api token to the environment variable 'TGTOKEN'"
    )
    exit(1)

engine = create_engine(config.get_db_uri('local_db/prototype_v1.db'))
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
        await message.reply(lexicon["en"]["hello"])
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
        await message.answer(lexicon['en']['retry_name'])
        return

    async with state.proxy() as data:
        data['user_name'] = message_text

    await state.set_state(GetBasicInfo.StartOfDay)
    await message.answer(lexicon['en']['basic_info_start_of_day'])


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
        await message.reply(lexicon['en']['retry_time'])
        return

    async with state.proxy() as data:
        data["end_time"] = time

        user = parse_user(
            user_id=message.from_id,
            user_name=data.get('user_name'),
            start_time=data.get('start_time'),
            end_time=data.get('end_time')
        )

    if user is None:
        logging.error(get_lexicon_with_argument('error', data.as_dict()))
        await message.answer(lexicon['en']['retry_trouble'])
        await state.finish()
        return

    try:
        repo = get_users_repo()
        repo.add_user(user)
    except Exception as e:
        logging.error(get_lexicon_with_argument('error', e.args))
        await message.answer(lexicon['en']['retry_trouble'])
        await state.finish()
        return

    await message.answer(lexicon['en']['write_success'])
    await state.finish()


@dp.message_handler(state="*", commands=["add_task"])
async def add_task(message: Message, state: FSMContext):

    repo = get_users_repo()
    user = repo.get_user_by_id(message.from_id)

    if user is None:
        """Please enter '/start'"""
        await message.answer(lexicon['en']['user_not_found'])
        return

    await state.set_state(GetTaskInfo.TaskNameState)
    await message.answer(lexicon["en"]["add_task"])


@dp.message_handler(state=GetTaskInfo.TaskNameState)
async def get_task_name(message: Message, state: FSMContext):
    """
    Used to parse beggining with gaining task name
    """
    message_text = message.text
    message_text = message_text.strip().lower()

    if not (1 <= len(message_text) <= 255):
        """Please retry"""
        await message.answer(lexicon['en']['retry_name'])

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

"""
@dp.message_handler(state=GetTaskInfo.TaskComplexityState)
async def get_task_complexity(message: Message, state: FSMContext):
    message_text = message.text
    result = parse_int(message_text)

    if result is None:
        \"\"\"Please retry\"\"\"
        await message.answer(lexicon["en"]["retry_int"])
        return

    async with state.proxy() as data:
        data["task_complexity"] = result

    await state.set_state(GetTaskInfo.TaskImportanceState)
    await message.answer(lexicon["en"]["get_importance_task"])
"""


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
                task_name=data.get('task_name'),
                duration=data.get('task_duration'),
                importance=data.get('task_importance'),
                start_time=data.get('task_start_time'),
                date=data.get('task_date')
            )
        print(task)

    if task is None:
        # Error
        logging.error(get_lexicon_with_argument('error', data.as_dict()))
        await message.answer(lexicon['en']['retry_trouble'])
        return

    try:
        repo.add_task(task)
        print(repo.get_active_tasks(message.from_id))
    except Exception as e:
        # Error
        logging.error(get_lexicon_with_argument('error', e.args))
        await message.answer(lexicon['en']['retry_trouble'])
        return

    print(repo.get_active_tasks(message.from_id))
    print(message.from_id)
    print(repo.get_task_by_id(task.task_id, message.from_id))

    await state.finish()
    await message.answer(lexicon["en"]["write_success"])


@dp.message_handler(state="*", commands=["add_event"])
async def add_event(message: Message, state: FSMContext):
    message_text = message.text

    repo = get_users_repo()
    user = repo.get_user_by_id(message.from_id)

    if user is None:
        """Please enter '/start'"""
        await message.answer(lexicon['en']['user_not_found'])
        return

    await state.set_state(GetEventInfo.EventNameState)
    await message.answer(lexicon["en"]["add_event"])


@dp.message_handler(state=GetEventInfo.EventNameState)
async def get_event_name(message: Message, state: FSMContext):
    message_text = message.text
    message_text = message_text.strip()

    if not (1 <= len(message_text) <= 255):
        """Please retry"""
        await message.answer(lexicon['en']['retry_name'])
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
        await message.answer(lexicon['en']['retry_int'])
        return

    async with state.proxy() as data:
        data['event_duration'] = result

    await state.set_state(GetEventInfo.EventDateState)
    await message.answer(lexicon['en']['event_date'])


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

    await state.set_state(state=GetEventInfo.EventRepeatEachNumberState)
    await message.answer(lexicon["en"]["event_repeat_each_number"])


@dp.message_handler(state=GetEventInfo.EventRepeatEachNumberState)
async def get_event_repeat_each_number(message: Message, state: FSMContext):
    message_text = message.text
    message_text = message_text.strip()

    if message_text.lower() == "no":
        """TODO: Get event, write event to db"""
        await message.answer(lexicon["en"]["write_success"])
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
            await message.answer(lexicon["en"]["retry_trouble"])
            await state.finish()
            return

        print("Dates:", dates)

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
            await message.answer(lexicon["en"]["retry_trouble"])
            await state.finish()
            return
    try:
        repo = get_events_repo()
        repo.add_many_events(events)
    except Exception as e:
        logging.error(get_lexicon_with_argument('error', e.args))
        await message.answer(lexicon['en']['retry_trouble'])
        return

    await state.finish()
    await message.answer(lexicon["en"]["write_success"])


@dp.message_handler(commands=["mark_history"])
async def mark_history(message: Message, state: FSMContext):
    message_text = message.text

    repo = get_users_repo()
    user = repo.get_user_by_id(message.from_id)

    if user is None:
        """Please enter '/start'"""
        await message.answer(lexicon['en']['user_not_found'])
        return

    repo = get_tasks_repo()

    async with state.proxy() as data:
        keyboard = await get_active_tasks_keyboard(repo.get_active_tasks(user.telegram_id))
        await message.answer(lexicon["en"]["mark_history"], reply_markup=keyboard)

    await state.set_state(MarkHistory.ChooseTask)


@dp.callback_query_handler(lambda c: "task_" in c.data, state=MarkHistory.ChooseTask)
async def mark_history_to_task(callback_query: CallbackQuery, state: FSMContext):
    message_text: str = callback_query.message.text
    task_id_str = callback_query.data.removeprefix('task_')
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
            await callback_query.message.answer(lexicon["en"]["marking_history_start_time"])

    await state.set_state(MarkHistory.MarkingHistoryStartTime)

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

    async with state.proxy() as data:
        data["duration"] = result

    await state.set_state(MarkHistory.MarkingHistoryIsDone)
    await message.answer(lexicon["en"]["marking_history_is_done"])


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
        await message.answer(lexicon['en']['retry_yesno'])
        return

    repo = get_tasks_repo()

    async with state.proxy() as data:
        data["is_done"] = result

        task: Task = data.get('marking_task')

        task.real_start = data.get('start_time')
        task.real_date = data.get('date', datetime.date.today())
        task.real_duration = data.get('duration')
        task.is_done = data.get('is_done')
        repo.add_task(task)

    await message.answer(lexicon['en']['write_success'])
    await state.finish()
