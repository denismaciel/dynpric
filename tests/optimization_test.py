import numpy as np
from dynpric.firms.ferreira2018 import find_optimal_price


prices = [29.9, 34.9, 39.9, 44.9]
demand = [0.8, 0.6, 0.3, 0.1]


def test_optimization():
    c = 0.25
    optimization_result = find_optimal_price(prices, demand, c)
    expected = np.array([0.0, 0.0, 0.75, 0.25])
    assert np.allclose(optimization_result.x, expected)

    c = 0.5
    optimization_result = find_optimal_price(prices, demand, c)
    expected = np.array([0.0, 2 / 3, 1 / 3, 0.0])
    assert np.allclose(optimization_result.x, expected)


def test_optimization_without_inventory_constraint():
    optimization_result = find_optimal_price(prices, demand)
    expected = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(optimization_result.x, expected)


if __name__ == '__main__':
    x = test_optimization_without_inventory_constraint()
