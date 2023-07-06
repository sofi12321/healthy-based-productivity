from datetime import datetime, timedelta


class Converter:
    def __init__(self, alpha):
        self.alpha = alpha

    def model_to_user(self, time, duration, offset,
                      current_date=datetime.now().replace(second=0, microsecond=0)):
        max_time = current_date + timedelta(minutes=self.alpha)
        task_date_user = time * (max_time - current_date) + current_date
        duration_user = int((duration * (max_time - current_date)).total_seconds() / 60)
        offset_user = int(offset * (max_time - current_date).total_seconds() / 60)

        # TODO: Uncomment only for debugging
        #         print("Current time:", current_date)
        #         print("Task: \n\t- time:", time, "\n\t- duration:", duration, "\n\t- offset:", offset)
        #         print("Alpha:", self.alpha, end = "\n\n")
        #         print("Task converted:", task_date_user)
        #         print("Duration converted:", duration_user)
        #         print("Offset converted:", offset_user)

        return task_date_user, duration_user, offset_user

    def user_to_model(self, task_date, duration, offset,
                      current_date=datetime.now().replace(second=0, microsecond=0)):
        max_time = current_date + timedelta(minutes=self.alpha)
        task_date_model = (task_date - current_date) / (max_time - current_date)
        duration_model = timedelta(minutes=duration) / (max_time - current_date)
        offset_model = timedelta(minutes=offset) / (max_time - current_date)

        # TODO: Uncomment only for debugging
        #         print("Current time:", current_date)
        #         print("Task: \n\t- time:", task_date, "\n\t- duration:", duration, "\n\t- offset:", offset)
        #         print("Alpha:", self.alpha, end = "\n\n")
        #         print("Task converted:", task_date_model)
        #         print("Duration converted:", duration_model)
        #         print("Offset converted:", offset_model)

        return task_date_model, duration_model, offset_model


if __name__ == '__main__':
    convertor = Converter(alpha=1440)
    convertor.model_to_user(time=0.5, duration=0.5, offset=0.5)
    convertor.user_to_model(task_date=datetime(2020, 12, 12, 12, 12), duration=60, offset=30)
