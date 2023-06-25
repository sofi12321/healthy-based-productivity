from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from domain.domain import Task


async def get_active_tasks_keyboard(active_tasks: [Task]) -> InlineKeyboardMarkup:
    keyboard = InlineKeyboardMarkup()

    for task in active_tasks:
        if task is None:
            continue

        keyboard = keyboard.add(
            InlineKeyboardButton(task.task_name, callback_data='task_[temp_id]')
        )

    if len(active_tasks) == 0 or active_tasks == [None]:
        keyboard = keyboard.add(
            InlineKeyboardButton("No active tasks", callback_data='blank')
        )

    return keyboard
