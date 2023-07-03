from entrypoints.telegram_bot_app import executor, dp

executor.start_polling(dp, skip_updates=True)
