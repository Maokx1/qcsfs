# SPC rules: https://en.wikipedia.org/wiki/Western_Electric_rules
# Unbiased sample standard deviation: https://en.wikipedia.org/wiki/Standard_deviation#Unbiased_sample_standard_deviation
# For better explanation check docs/SPC.md
from math import sqrt
import statistics

import matplotlib.pyplot as plt


def check_spc_rules(data: list[int] | list[float], sigma_lines: list[int] | list[float],
                    is_xbar: bool = True) -> list[list[int]]:
    """
    This function checks if any point violates the Western Electric zone rules.
    If so, it returns the index of that point in the list.
    #### Args:
        data (list[int | float]): A list of data points.
        sigma_lines (list[int | float]): A list of points marking limits e.g. ±𝜎, ±2𝜎, ±3𝜎
        is_xbar (bool): If true, more Western Electric zone rules will be checked.
    #### Returns:
        (list[list[int | float]]): A list of indexes of data points that violate the rules.
    #### Raises:
        IndexError: When either the date or sigma list is empty.
    """
    if not data:
        raise IndexError
    if not sigma_lines:
        raise IndexError
    # 1. Any single data point falls outside the 3𝜎-limit from the centerline.
    sigma_lines.sort(reverse=True)
    marked_values = [[idx for idx, x in enumerate(data)
                      if x > sigma_lines[0] or x < sigma_lines[-1]]]
    if is_xbar:
        # 2. Two out of three consecutive points fall beyond the 2𝜎-limit, on the same side of the centerline.
        mp = set()
        for idx in range(len(data) - 2):
            if len(pts := [idx+i for i, p in enumerate(data[idx:idx + 3]) if p > sigma_lines[1]]) >= 2:
                mp.update(pts)
            if len(pts := [idx+i for i, p in enumerate(data[idx:idx + 3]) if p < sigma_lines[-2]]) >= 2:
                mp.update(pts)
        marked_values.append(list(mp))
        # 3. Four out of five consecutive points fall beyond the 1𝜎-limit, on the same side of the centerline.
        mp.clear()
        for idx in range(len(data) - 4):
            if len(pts := [idx+i for i, p in enumerate(data[idx:idx + 5]) if p > sigma_lines[2]]) >= 4:
                mp.update(pts)
            if len(pts := [idx+i for i, p in enumerate(data[idx:idx + 5]) if p < sigma_lines[-3]]) >= 4:
                mp.update(pts)
        marked_values.append(list(mp))
        # 4. 9 consecutive points fall on the same side of the centerline.
        mp.clear()
        for idx in range(len(data) - 9):
            if len(pts := [idx+i for i, p in enumerate(data[idx:idx + 10]) if p > sigma_lines[3]]) >= 9:
                mp.update(pts)
            if len(pts := [idx+i for i, p in enumerate(data[idx:idx + 10]) if p < sigma_lines[3]]) >= 9:
                mp.update(pts)
        marked_values.append(list(mp))
    return marked_values


def create_xbar_chart(data: list[int] | list[float], num_of_subgroups: int = 32,
                      show_parameters: bool = False) -> tuple:
    """
    This function creates X-bar control chart based on given data.
    #### Args:
        data (list[int | float]): A list of data points.
        num_of_subgroups (int): Number of subgroups into which the points are to be divided.
        show_parameters (bool): If true, prints parameters such as: mean, UCL, LCL and standard deviation. 
    #### Returns:
        (tuple): Returns parameters such as: subgroup size, means of subgroups, mean of means, standard deviations of subgroups etc.
    """
    num_of_screws = len(data)
    # Due to integer division the last subgroup may be smaller in size compared to the other subgroups.
    subgroup_size = num_of_screws // num_of_subgroups
    subgroups = [data[n:n + subgroup_size]
                for n in range(0, num_of_screws, subgroup_size)]     
    means_of_subgroups = [statistics.mean(subgroup)
                          for subgroup in subgroups]
    xbar = statistics.mean(means_of_subgroups)
    standard_deviations_of_subgroups = [statistics.stdev(subgroup)
                                        for subgroup in subgroups]
    sbar = statistics.mean(standard_deviations_of_subgroups)

    # Unbiased estimation of standard deviation is dependent on parameter c4, which depends on the gamma function,
    # but in this programme the function (4 * n - 4) / (4 * n - 3) is good enough to approximate c4.
    # To be precise, the absolute error between the above function and c4 is:
    # for n < 5 absolute error is 0.2%,
    # for n = 5 absolute error is 0.1%,
    # for n >= 10 absolute error is 0.03%,
    if subgroup_size == 2:
        a = 0.79788
    elif 2 < subgroup_size < 5:
        a = 0.9
    else:
        a = (4 * subgroup_size - 4) / (4 * subgroup_size - 3)
    c4 = sqrt(2 / (subgroup_size - 1)) * a
    true_standard_deviation = sbar / c4

    # Calculating 𝜎 limits
    xbar_ucl = xbar + 3 * (true_standard_deviation / sqrt(subgroup_size))
    xbar_lcl = xbar - 3 * (true_standard_deviation / sqrt(subgroup_size))

    xbar_2op = xbar + 2 * (true_standard_deviation / sqrt(subgroup_size))
    xbar_2om = xbar - 2 * (true_standard_deviation / sqrt(subgroup_size))

    xbar_1op = xbar + (true_standard_deviation / sqrt(subgroup_size))
    xbar_1om = xbar - (true_standard_deviation / sqrt(subgroup_size))
    s_lines = [xbar_ucl, xbar_2op, xbar_1op,
               xbar, xbar_1om, xbar_2om, xbar_lcl]
    marked_values = check_spc_rules(means_of_subgroups, s_lines)
    # Displaying X-bar control chart
    plt.figure('X-bar chart')
    plt.title(f'X-bar control chart')
    plt.xlabel('Number of subgroup')
    plt.ylabel(f'Mean of given subgroup (size = {subgroup_size})')
    # Displaying control limits
    plt.axhline(y=xbar, color='r', linestyle='-', linewidth=1)
    plt.axhline(y=xbar_ucl, color='y', linestyle='-', linewidth=1)
    plt.axhline(y=xbar_lcl, color='y', linestyle='-', linewidth=1)
    plt.axhline(y=xbar_2op, color='k', linestyle='-', linewidth=1)
    plt.axhline(y=xbar_2om, color='k', linestyle='-', linewidth=1)
    plt.axhline(y=xbar_1op, color='b', linestyle='-', linewidth=1)
    plt.axhline(y=xbar_1om, color='b', linestyle='-', linewidth=1)
    plt.plot(means_of_subgroups, color='b', linestyle='-',
             marker='o', linewidth=3)
    # Marking points violating SPC rules
    plt.plot(means_of_subgroups, color='c', markevery=marked_values[3],
             linestyle='None', marker='o', label='Rule #4')
    plt.plot(means_of_subgroups, color='g', markevery=marked_values[2],
             linestyle='None', marker='o', label='Rule #3')
    plt.plot(means_of_subgroups, color='m', markevery=marked_values[1],
             linestyle='None', marker='o', label='Rule #2')
    plt.plot(means_of_subgroups, color='r', markevery=marked_values[0],
             linestyle='None', marker='o', label='Rule #1')
    plt.legend()
    if show_parameters:
        print(f'Centerline (mean of means): {xbar:.4f}')
        print(f'UCL: {xbar_ucl:.4f}\nLCL: {xbar_lcl:.4f}')
        print(f'Standard deviation (sigma): {true_standard_deviation:.4f}')
    return subgroup_size, means_of_subgroups, xbar, standard_deviations_of_subgroups, \
        sbar, c4, true_standard_deviation


def create_xbar_s_chart(data: list[int] | list[float], num_of_subgroups: int = 32, show_parameters: bool = False):
    """
    This function creates Xbar-S control chart based on given data.
    #### Args:
        data (list[int | float]): A list of data points.
        num_of_subgroups (int): Number of subgroups into which the points are to be divided.
        show_parameters (bool): If true, prints parameters such as: mean, UCL, LCL, standard deviation. 
    """
    subgroup_size, _, xbar, standard_deviations_of_subgroups, sbar, \
        c4, true_standard_deviation = create_xbar_chart(data, num_of_subgroups=num_of_subgroups,
                                                        show_parameters=False)
    s_ucl = sbar + 3 * true_standard_deviation * sqrt(1 - c4 ** 2)
    if lcl := sbar - 3 * true_standard_deviation * sqrt(1 - c4 ** 2) < 0:
        s_lcl = 0
    else:
        s_lcl = lcl

    marked_values = check_spc_rules(standard_deviations_of_subgroups,
                                    [s_ucl, s_lcl], False)
    # Display S control chart
    plt.figure('S chart')
    plt.title(f'S control chart')
    plt.xlabel('Number of subgroup')
    plt.ylabel(f'Standard deviation of subgroup (size = {subgroup_size})')
    # Displaying control limits
    plt.axhline(y=sbar, color='g', linestyle='-', linewidth=1)
    plt.axhline(y=s_ucl, color='r', linestyle='-', linewidth=1)
    plt.axhline(y=s_lcl, color='r', linestyle='-', linewidth=1)
    plt.plot(standard_deviations_of_subgroups, color='b',
             linestyle='-', marker='o', linewidth=3)
    # Marking points violating SPC rules
    plt.plot(standard_deviations_of_subgroups, color='m',
             markevery=marked_values[0], marker='o', linestyle='None')
    if show_parameters:
        print(f'Centerline (mean of means): {xbar:.4f}')
        print(f'Standard deviation UCL: {s_ucl:.4f}\n'
              f'Standard deviation LCL: {s_lcl:.4f}')
        print(f'Standard deviation (sigma): {true_standard_deviation:.4f}')
