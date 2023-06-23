from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor

from states.states import GetBasicInfo, BaseState
from lexicon.lexicon import lexicon
from utils.utils import parse_time

import os
import logging

API_TOKEN: str = os.getenv("TGTOKEN")

if API_TOKEN is None:
    logging.error("Please enter telegram bot api token to the environment variable 'TGTOKEN'")
    exit(1)

bot: Bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp: Dispatcher = Dispatcher(bot, storage=storage)


@dp.message_handler(state='*', commands="start")
async def process_start_command(message: Message, state: FSMContext):
    data = await state.get_data()
    if 'start_time' in data and 'end_time' in data:
        await message.reply(lexicon['en']['hello'])
        return

    await state.set_state(GetBasicInfo.StartEndOfDay)
    logging.info("Getting info about start and end of time from user")
    await message.reply(lexicon['en']['first_hello'])


@dp.message_handler(state=GetBasicInfo.StartEndOfDay)
async def get_start_end_day(message: Message, state: FSMContext):
    message_text = message.text
    times = parse_time(message_text)

    if times is None:
        await message.reply(lexicon['en']['retry_first_hello'])
        return

    async with state.proxy() as data:
        data['start_time'] = times[0]
        data['end_time'] = times[1]

    await state.finish()
    await message.reply(lexicon['en']['write_success'])


@dp.message_handler(state='*', commands=["add_task"])
async def add_task(message: Message):
    # TODO: checking user for providing necessary info

    pass


@dp.message_handler(commands=["add_event"])
async def add_event(message: Message):
    pass


@dp.message_handler(commands=["mark_history"])
async def mark_history(message: Message):
    pass