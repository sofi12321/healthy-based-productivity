import csv
import random
from datetime import datetime, timedelta


class GeneratorOfSchedule:
    def __init__(self, num_days, filename="schedule_gen.csv"):
        """
        Initialize the GeneratorOfSchedule object
        :param num_days: number of days for which the schedule will be generated
        :param filename: name of the file to which the schedule will be saved
        """
        self.num_days = num_days
        self.filename = filename
        self.schedule = []

    def _generate_start_time(self):
        """
        Generate a random start time of the day in the given range (06:00 - 09:30)
            from which the schedule will be generated
        :return: start time
        """
        start_time = datetime.strptime("06:00", "%H:%M") + timedelta(minutes=random.randrange(0, 225, 5))
        return timedelta(hours=start_time.hour, minutes=start_time.minute)

    def _generate_next_start_time(self, curr_day):
        """
        Generate a random start time of the next day based on the date of the current day
            and assuming that the user sleeps 7-9 hours
        :param curr_day: start time of the previous day
        :return: end time
        """
        curr_day = curr_day.replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = curr_day + timedelta(hours=24) + self._generate_start_time()
        return start_time

    def _generate_label_num(self):
        """
        Generate a random label of the activity based on the following corresponding numbers:
            0 - Daily Routine
            1 - Passive Rest
            2 - Physical Activity
            3 - Work-study
        :return: number of the label
        """
        return random.randint(0, 3)

    def _generate_duration(self, label_num):
        """
        Generate a random duration of the activity based on the given ranges
            assuming that Work-study activities are the longest
            and Daily Routine activities are the shortest
        :param label_num: number of the label
        :return: duration
        """
        if label_num == 3:
            return random.randrange(60, 190, 10)
        elif label_num == 0:
            return random.randrange(5, 40, 5)
        else:
            return random.randrange(20, 105, 5)

    def _generate_importance(self, label_num):
        """
        Generate a random importance based on the given range (0 - 3)
            assuming that Work-study activities are the most important
        :param label_num: number of the label
        :return: importance
        """
        if label_num == 3:
            return random.choices([1, 2, 3], weights=[0.1, 0.35, 0.55])[0]
        else:
            return random.choices([0, 1, 2, 3], weights=[0.5, 0.3, 0.15, 0.05])[0]

    def generate_schedule(self):
        """
        Generate a schedule for the given number of days
        """
        curr_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + \
                    timedelta(days=1) + self._generate_start_time()
        next_start_time = self._generate_next_start_time(curr_time)
        day_end_time = next_start_time - timedelta(hours=random.randint(7, 9))

        for _ in range(self.num_days):
            num_activities = random.randint(8, 15)
            activities = []

            for i in range(num_activities):
                if curr_time <= day_end_time:
                    # Generate random activity details
                    label_num = self._generate_label_num()
                    while i > 0 and label_num == activities[-1]["label_num"]:
                        if random.random() < 0.75:
                            label_num = self._generate_label_num()
                        else:
                            break
                    duration = self._generate_duration(label_num)
                    importance = self._generate_importance(label_num)

                    # Add activity to the schedule
                    activities.append({
                        "label_num": label_num,
                        "duration": duration,
                        "importance": importance,
                        "start_time": curr_time.strftime("%H:%M"),
                        "date": curr_time.strftime("%d/%m/%Y")
                    })

                    # Update current time
                    curr_time += timedelta(minutes=duration)
                    curr_time += timedelta(minutes=random.randrange(10, 180, 5))
                else:
                    break

            self.schedule.extend(activities)
            curr_time = next_start_time
            next_start_time = self._generate_next_start_time(curr_time)
            day_end_time = next_start_time - timedelta(hours=random.randint(7, 9))

        self._write_to_csv()

    def _write_to_csv(self):
        """
        Write the generated schedule to a csv file
        """
        header = ["Label Number", "Duration", "Importance", "Start Time", "Date"]

        with open(self.filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for activity in self.schedule:
                writer.writerow([activity["label_num"], activity["duration"], activity["importance"],
                                 activity["start_time"], activity["date"]])

        print(f"Schedule generated and saved in '{self.filename}'.")


# TODO: Uncomment only for debugging
if __name__ == "__main__":
    generator = GeneratorOfSchedule(10000)
    generator.generate_schedule()
