from tg_bot.entrypoints.telegram_bot_app import executor, dp


def run():
    executor.start_polling(dp, skip_updates=True)
