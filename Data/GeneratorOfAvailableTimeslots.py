import random
import numpy as np
from fractions import Fraction


class GeneratorOfAvailableTimeslots:
    def __init__(self, number_of_available_timeslots, start_time=0.0, end_time=1.0):
        self.number_of_available_timeslots = number_of_available_timeslots
        self.start = start_time
        self.end = end_time
        self.available_timeslots = np.empty(number_of_available_timeslots, dtype=np.ndarray)

    def _generate_inner_intervals(self):
        num_intervals = random.choices(range(1, 18), weights=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 1])[0]
        intervals = np.empty((num_intervals, 2))

        intervals_sum = 0.0  # Track the cumulative sum of interval sizes
        for i in range(num_intervals):
            remaining_space = (self.end - self.start) - intervals_sum
            interval_size = random.uniform(0, remaining_space)
            lower_bound = self.start + intervals_sum
            upper_bound = lower_bound + interval_size

            # Convert lower and upper bounds to fractions
            lower_fraction = Fraction(lower_bound).limit_denominator()
            upper_fraction = Fraction(upper_bound).limit_denominator()

            intervals[i] = [float(lower_fraction), float(upper_fraction)]  # Convert to decimal fractions

            if upper_fraction == 1 and i < num_intervals - 1:
                return intervals[:i+1]  # Return the intervals generated so far

            intervals_sum += interval_size  # Update the cumulative sum
            # print(intervals.shape)

        return intervals

    def generate_available_timeslots(self):
        for i in range(self.number_of_available_timeslots):
            self.available_timeslots[i] = self._generate_inner_intervals()
        return self.available_timeslots

    def print_available_timeslots(self):
        for i, interval in enumerate(self.available_timeslots):
            print(f"Interval {i+1}: {interval}")


# TODO: Uncomment only for debugging
# if __name__ == '__main__':
#     generator = GeneratorOfAvailableTimeslots(5)
#     generator.generate_available_timeslots()
#     generator.print_available_timeslots()
