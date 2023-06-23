from aiogram.dispatcher.filters.state import State, StatesGroup


class GetBasicInfo(StatesGroup):
    StartEndOfDay = State()


class BaseState(StatesGroup):
    Unit = State()
