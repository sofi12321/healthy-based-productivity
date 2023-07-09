from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from tg_bot.domain.domain import Task, Event
from typing import Optional


async def get_active_tasks_keyboard(active_tasks: [Task]) -> InlineKeyboardMarkup:
    keyboard = InlineKeyboardMarkup()

    for task in active_tasks:
        if task is None:
            continue

        keyboard = keyboard.add(
            InlineKeyboardButton(task.task_name, callback_data=f'task_{task.task_id}')
        )

    if len(active_tasks) == 0 or active_tasks == [None]:
        keyboard = keyboard.add(
            InlineKeyboardButton("No active tasks", callback_data='task_notask')
        )

    return keyboard


async def get_start_buttons_keyboard() -> ReplyKeyboardMarkup:
    keyboard_buttons_text = ['+task', '+event', 'mark history', 'plan']
    buttons = list(map(KeyboardButton, keyboard_buttons_text))

    print(buttons)
    keyboard: ReplyKeyboardMarkup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)

    keyboard = keyboard.row(*buttons)

    return keyboard


async def get_back_button_keyboard(keyboard: Optional[ReplyKeyboardMarkup] = None) -> ReplyKeyboardMarkup:
    if keyboard is None:
        keyboard: ReplyKeyboardMarkup = ReplyKeyboardMarkup()

    return keyboard.add(
        InlineKeyboardButton("Back", callback_data='listtaskback')
    )


async def get_tasks_events_keyboard(tasks: [Task], events: [Event]) -> InlineKeyboardMarkup:
    keyboard = InlineKeyboardMarkup()

    pass