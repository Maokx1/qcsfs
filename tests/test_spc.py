import pytest
import statistics

from . import (
    check_spc_rules,
    create_xbar_chart,
    create_xbar_s_chart
)


def test_check_spc_rules_no_valid_input():
    with pytest.raises(IndexError):
        check_spc_rules([], [1])
        # check_spc_rules([1], [])


@pytest.mark.parametrize('rule_num, data, sigma, index', [
    (0, [1, 1, 1, 1, 1, 1, 50], [42.73, 30.86,
     18.99, 7.12, -4.74, -16.61, -28.48], 6),  # 1st rule
    (0, [1, 1, 1, 1, 1, 1, -50], [31.68, 19.33,
     6.98, -5.38, -17.73, -30.08, -42.43], 6),  # 1st rule
    (1, [1, 1, 1, 1, 1, 1, 1, 32, 32], [
     36.08, 26.68, 17.29, 7.89, -1.51, -10.91, -20.3], 7),  # 2nd rule
    (1, [1, 1, 1, 1, 1, 1, 1, -32, -32],
     [23.68, 13.68, 3.67, -6.33, -16.34, -26.34, -36.35], 7),  # 2nd rule
    (2, [1, 1, 1, -4, -2, -1, 1, 5, 5, 5, 5, 1, -3],
     [7.67, 5.5, 3.33, 1.15, -1.02, -3.19, -5.36], 7),  # 3rd rule
    (2, [1, 1, 1, -4, -2, -1, 1, -5, -5, -5, -5, 1, -3],
     [3.68, 1.81, -0.06, -1.92, -3.79, -5.66, -7.52], 7),  # 3rd rule
    (3, [0, 1, 2, 2.5, 1, 1, 1.5, 1, 1, 1, 0.5, -1, -0.6, 1],
     [2.77, 2.13, 1.49, 0.85, 0.21, -0.43, -1.07], 7),  # 4th rule
    (3, [0, -1, -2, -2.5, -1, -1, -1.5, -1, -1, -1, -0.5, 1, 0.6, -1],
     [1.07, 0.43, -0.21, -0.85, -1.49, -2.13, -2.77], 7),  # 4th rule
])
def test_check_spc_rules_marking(rule_num, data, sigma, index):
    assert index in check_spc_rules(data=data, sigma_lines=sigma,
                                    is_xbar=True)[rule_num]


def test_crate_xbar_chart_no_data():
    with pytest.raises(ValueError):
        create_xbar_chart([])


def test_create_xbar_chart_too_small_data():
    with pytest.raises(statistics.StatisticsError, match='at least two data points'):
        create_xbar_chart([1], num_of_subgroups=1)


def test_create_xbar_chart():
    data = [1, 1, 1, 1]
    ans = (2, [1.0, 1.0], 1.0, [0.0, 0.0], 0.0, 1.1283727171462452, 0.0)
    assert create_xbar_chart(data=data, num_of_subgroups=2)


def test_create_xbar_s_chart_no_data():
    with pytest.raises(ValueError):
        create_xbar_s_chart([])


def test_create_xbar_s_chart_too_small_data():
    with pytest.raises(statistics.StatisticsError):
        create_xbar_s_chart([1], num_of_subgroups=1)
