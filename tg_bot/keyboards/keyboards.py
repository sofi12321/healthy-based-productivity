from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton
)
from tg_bot.domain.domain import Task, Event
from typing import Optional, Union, List


async def get_active_tasks_keyboard(active_tasks: [Task]) -> InlineKeyboardMarkup:
    keyboard = InlineKeyboardMarkup()

    for task in active_tasks:
        if task is None:
            continue

        keyboard = keyboard.add(
            InlineKeyboardButton(task.task_name, callback_data=f"task_{task.task_id}")
        )

    if len(active_tasks) == 0 or active_tasks == [None]:
        keyboard = keyboard.add(
            InlineKeyboardButton("No active tasks", callback_data="task_notask")
        )

    return keyboard


async def get_start_buttons_keyboard() -> ReplyKeyboardMarkup:
    keyboard_buttons_text = ["📌 Add Task", "📆 Add Event", "✅ Mark History", "📝 Plan", "📜 List", "🪄 Formats"]

    buttons = list(map(KeyboardButton, keyboard_buttons_text))

    keyboard: ReplyKeyboardMarkup = ReplyKeyboardMarkup(
        resize_keyboard=True, one_time_keyboard=True
    )

    keyboard = keyboard.row(buttons[0], buttons[3])
    keyboard = keyboard.row(buttons[1], buttons[4])
    keyboard = keyboard.row(buttons[2], buttons[5])

    return keyboard


async def add_back_button_keyboard(
    keyboard: Optional[InlineKeyboardMarkup] = None,
) -> InlineKeyboardMarkup:
    if keyboard is None:
        keyboard = InlineKeyboardMarkup()

    return keyboard.add(InlineKeyboardButton("Back", callback_data="listitemback"))


async def add_cancel_button_keyboard(
    keyboard: Optional[InlineKeyboardMarkup]=None,
) -> InlineKeyboardMarkup:
    if keyboard is None:
        keyboard = InlineKeyboardMarkup()

    return keyboard.add(
        InlineKeyboardButton("Return to Main", callback_data="listitemreturn")
    )


async def get_tasks_events_keyboard(
    task_event_arr: List[Union[Task, Event]]
) -> InlineKeyboardMarkup:
    keyboard = InlineKeyboardMarkup()

    for item in task_event_arr:
        if isinstance(item, Event):
            keyboard = keyboard.add(
                InlineKeyboardButton(
                    item.event_name, callback_data=f"listitem_event_{item.event_id}"
                )
            )

        if isinstance(item, Task):
            keyboard = keyboard.add(
                InlineKeyboardButton(
                    item.task_name, callback_data=f"listitem_task_{item.task_id}"
                )
            )

    return keyboard


async def add_delete_list_button(
    task_event: Union[Task, Event],
    keyboard: Optional[InlineKeyboardMarkup] = None,
) -> InlineKeyboardMarkup:
    if not isinstance(task_event, Task) and not isinstance(task_event, Event):
        return None

    if keyboard is None:
        keyboard = InlineKeyboardMarkup()

    if isinstance(task_event, Event):
        type_str = "event"
        item_id = task_event.event_id

    else:
        type_str = "task"
        item_id = task_event.task_id

    return keyboard.add(InlineKeyboardButton("Delete", callback_data=f"listitemdelete_{type_str}_{item_id}"))