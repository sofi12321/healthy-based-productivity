import random
import numpy as np
from fractions import Fraction


class GeneratorOfAvailableTimeslots:
    def __init__(self, number_of_available_timeslots, start_time=0.0, end_time=1.0):
        """
        Initialize the generator of available timeslots
        :param number_of_available_timeslots: number of available timeslots samples to generate
        :param start_time: start time of the day expressed as a decimal fraction from 0 to 1
        :param end_time: end time of the day expressed as a decimal fraction from 0 to 1
        """
        self.number_of_available_timeslots = number_of_available_timeslots
        self.start = start_time
        self.end = end_time
        self.available_timeslots = np.empty(number_of_available_timeslots, dtype=np.ndarray)

    def _generate_inner_intervals(self):
        """
        Generate a random number of intervals in one sample of the available timeslots
        :return: sample of the available timeslots
        """
        # Choose the number of intervals based on the given weights
        num_intervals = random.choices(range(1, 18), weights=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 1])[0]
        # Initialize the array of intervals
        intervals = np.empty((num_intervals, 2))

        # Track the cumulative sum of interval sizes
        intervals_sum = 0.0

        for i in range(num_intervals):
            # Calculate the remaining space
            remaining_space = (self.end - self.start) - intervals_sum

            # Generate a random interval size
            if random.random() < 0.7:
                interval_size = random.uniform(0, remaining_space / 16)
            else:
                interval_size = random.uniform(0, remaining_space / 2)

            # Generate a random shift
            shift = random.uniform(0, interval_size / 2)
            interval_size -= shift

            # Shift the lower bound or the upper bound of the interval
            if random.random() < 0.5:
                lower_bound = self.start + intervals_sum + shift
                upper_bound = lower_bound + interval_size
            else:
                lower_bound = self.start + intervals_sum
                upper_bound = lower_bound + interval_size

            # Convert lower and upper bounds to fractions
            lower_fraction = Fraction(lower_bound).limit_denominator()
            upper_fraction = Fraction(upper_bound).limit_denominator()

            # Convert to decimal fractions
            intervals[i] = [float(lower_fraction), float(upper_fraction)]

            # Return the intervals generated so far except the last one if the upper bound is greater than 1
            if upper_fraction > 1:
                return intervals[:i]
            # Return the intervals generated so far if the upper bound is 1
            elif upper_fraction == 1 and i < num_intervals - 1:
                return intervals[:i+1]

            # Update the cumulative sum
            intervals_sum += interval_size + shift + random.uniform(0, remaining_space / 2)

        return intervals

    def generate_available_timeslots(self):
        """
        Generate the available timeslots
        :return:
        """
        for i in range(self.number_of_available_timeslots):
            self.available_timeslots[i] = self._generate_inner_intervals()
        return self.available_timeslots

    def print_available_timeslots(self):
        """
        Print the generated available timeslots
        :return: None
        """
        for i, interval in enumerate(self.available_timeslots):
            print(f"Interval {i+1}: {interval}")


# TODO: Uncomment only for debugging
# if __name__ == '__main__':
#     generator = GeneratorOfAvailableTimeslots(100)
#     generator.generate_available_timeslots()
#     generator.print_available_timeslots()

#%%
