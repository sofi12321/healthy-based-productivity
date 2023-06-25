from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor

from states.states import GetBasicInfo, BaseState, GetTaskInfo
from lexicon.lexicon import lexicon
from utils.utils import (
    start_end_of_day_time,
    parse_task,
    parse_int,
    parse_time,
    parse_date,
    get_task,
)
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
    times = start_end_of_day_time(message_text)

    if times is None:
        await message.reply(lexicon["en"]["retry_first_hello"])
        return

    async with state.proxy() as data:
        data["start_time"] = times[0]
        data["end_time"] = times[1]

    await state.finish()
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
        await state.finish()
        await message.answer(lexicon["en"]["write_success"])
        return

    result = parse_date(message_text)

    if result is None:
        """Please retry"""
        await message.answer(lexicon["en"]["retry_optional_date"])
        return

    async with state.proxy() as data:
        data["task_date"] = result
        data["last_date"] = get_task(data)

    await state.finish()
    await message.answer(lexicon["en"]["write_success"])


@dp.message_handler(commands=["add_event"])
async def add_event(message: Message):
    pass


@dp.message_handler(commands=["mark_history"])
async def mark_history(message: Message):
    pass
