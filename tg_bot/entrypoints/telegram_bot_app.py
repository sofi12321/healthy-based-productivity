from aiogram import Bot, Dispatcher
from aiogram.types import Message, CallbackQuery
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor

from states.states import (
    GetBasicInfo,
    BaseState,
    GetTaskInfo,
    GetEventInfo,
    MarkHistory,
    UnitState,
)
from lexicon.lexicon import lexicon, get_lexicon_with_argument
from utils.utils import (
    parse_start_end_of_day_time,
    parse_task,
    parse_int,
    parse_time,
    parse_date,
    get_task,
    parse_event,
    parse_marking_history,
    parse_repeated_arguments,
    parse_int_in_range,
    check_repeat_each_argument,
)
from keyboards.keyboards import get_active_tasks_keyboard
from domain.domain import Task

import os
import logging

API_TOKEN: str = os.getenv("TGTOKEN")

if API_TOKEN is None:
    logging.error(
        "Please enter telegram bot api token to the environment variable 'TGTOKEN'"
    )
    exit(1)

bot: Bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp: Dispatcher = Dispatcher(bot, storage=storage)


@dp.message_handler(state="*", commands="start")
async def process_start_command(message: Message, state: FSMContext):
    data = await state.get_data()
    if "start_time" in data and "end_time" in data:
        await message.reply(lexicon["en"]["hello"])
        return

    await state.set_state(GetBasicInfo.StartEndOfDay)
    logging.info("Getting info about start and end of time from user")
    await message.answer(lexicon["en"]["first_hello"])


@dp.message_handler(state=GetBasicInfo.StartEndOfDay)
async def get_start_end_day(message: Message, state: FSMContext):
    message_text = message.text
    times = parse_start_end_of_day_time(message_text)

    if times is None:
        await message.reply(lexicon["en"]["retry_first_hello"])
        return

    async with state.proxy() as data:
        data["start_time"] = times[0]
        data["end_time"] = times[1]

    await state.set_state(UnitState)
    await message.answer(lexicon["en"]["write_success"])


@dp.message_handler(state="*", commands=["add_task"])
async def add_task(message: Message, state: FSMContext):
    # TODO: checking user for providing necessary info
    await state.set_state(GetTaskInfo.TaskNameState)
    await message.answer(lexicon["en"]["add_task"])


@dp.message_handler(state=GetTaskInfo.TaskNameState)
async def get_task_name(message: Message, state: FSMContext):
    """
    Used to fully parse task in specific format
    or just gain task name
    """
    message_text = message.text

    result = parse_task(message_text)
    if result is not None:
        task_name, duration, importance, complexity, start_time, date = result
        async with state.proxy() as data:
            task = Task(
                task_name=task_name,
                duration=duration,
                importance=importance,
                complexity=complexity,
                start_time=start_time,
                date=date,
            )
            data["last_task"] = task
            print(data)
        await state.set_state(UnitState)
        await message.answer(lexicon["en"]["write_success"])
        return

    async with state.proxy() as data:
        data["task_name"] = message_text

    await state.set_state(GetTaskInfo.TaskDurationState)
    await message.answer(lexicon["en"]["get_duration_task"])


@dp.message_handler(state=GetTaskInfo.TaskDurationState)
async def get_task_duration(message: Message, state: FSMContext):
    message_text = message.text
    result = parse_int(message_text)

    if result is None:
        """Please retry"""
        await message.answer(lexicon["en"]["retry_int"])
        return

    async with state.proxy() as data:
        data["task_duration"] = result

    await state.set_state(GetTaskInfo.TaskComplexityState)
    await message.answer(lexicon["en"]["get_complexity_task"])


@dp.message_handler(state=GetTaskInfo.TaskComplexityState)
async def get_task_complexity(message: Message, state: FSMContext):
    message_text = message.text
    result = parse_int(message_text)

    if result is None:
        """Please retry"""
        await message.answer(lexicon["en"]["retry_int"])
        return

    async with state.proxy() as data:
        data["task_complexity"] = result

    await state.set_state(GetTaskInfo.TaskImportanceState)
    await message.answer(lexicon["en"]["get_importance_task"])


@dp.message_handler(state=GetTaskInfo.TaskImportanceState)
async def get_task_duration(message: Message, state: FSMContext):
    message_text = message.text
    result = parse_int(message_text)

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
async def get_task_start_time(message: Message, state: FSMContext):
    message_text = message.text

    if message_text == "no":
        async with state.proxy() as data:
            data["last_task"] = get_task(data)
        await state.set_state(UnitState)
        await message.answer(lexicon["en"]["write_success"])
        return

    result = parse_date(message_text)

    if result is None:
        """Please retry"""
        await message.answer(lexicon["en"]["retry_optional_date"])
        return

    async with state.proxy() as data:
        data["task_date"] = result
        data["last_task"] = get_task(data)

    await state.set_state(UnitState)
    await message.answer(lexicon["en"]["write_success"])


@dp.message_handler(state="*", commands=["add_event"])
async def add_event(message: Message, state: FSMContext):
    message_text = message.text
    await state.set_state(GetEventInfo.EventNameState)
    await message.answer(lexicon["en"]["add_event"])


@dp.message_handler(state=GetEventInfo.EventNameState)
async def get_event_name(message: Message, state: FSMContext):
    message_text = message.text

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
        """Please retry, TODO"""
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

    await state.finish()
    await message.answer(lexicon["en"]["write_success"])
    """
    TODO:
        -add to sql database
    """


@dp.message_handler(commands=["mark_history"])
async def mark_history(message: Message, state: FSMContext):
    message_text = message.text

    await state.set_state(MarkHistory.ChooseTask)

    async with state.proxy() as data:
        keyboard = await get_active_tasks_keyboard([data.get("last_task")])
        await message.answer(lexicon["en"]["mark_history"], reply_markup=keyboard)


@dp.callback_query_handler(lambda c: "task_" in c.data, state=MarkHistory.ChooseTask)
async def mark_history_to_task(callback_query: CallbackQuery, state: FSMContext):
    message_text: str = callback_query.message.text
    async with state.proxy() as data:
        task = data.get("last_task")
        if task is None:
            await callback_query.answer("No tasks")
        else:
            await callback_query.message.answer(
                f"""Task:
                Task name: {task.task_name}\t
                Task duration: {task.duration}\t
                Task importance: {task.importance}\t
                Task complexity: {task.complexity}\t
                Task start time: {task.start_time or "No time"}\t
                Task date: {task.date or "No date"}\n"""
            )
            await callback_query.message.answer(lexicon["en"]["ask_history"])

    await state.set_state(MarkHistory.MarkingHistory)

    async with state.proxy() as data:
        data["marking_id"] = callback_query.data


@dp.message_handler(state=MarkHistory.MarkingHistory)
async def marking_history(message: Message, state: FSMContext):
    message_text = message.text

    result = parse_marking_history(message_text)

    if result is None:
        await message.answer(lexicon["en"]["retry_history"])
        return

    async with state.proxy() as data:
        data[f"marked_history_{data.get('marking_id')}"] = result

    await state.set_state(UnitState)
    await message.answer(lexicon["en"]["write_success"])
