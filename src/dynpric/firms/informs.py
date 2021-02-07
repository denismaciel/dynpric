import random
from typing import List

import numpy as np
import sklearn.linear_model


class OLSFirm:
    """
    - Other competitor's actions are random. Modeled as random noise.
    - 40 periods: exploration
    - 960 periods: 94% of the time: regular exploitation; 5% exploration;
    1% competitive disruption
    - Exploration
        * Price randomly sampled from (0, 100)
    - Exploitation
        * Four models
            * q ~ p
            * q ~ log p
            * log q ~ p
            * log q ~ log p
        * Chooses model with highest R2
        * Optimize through line search
        * Add small perturbation for further exploration
    - Competitive disruption
        * Set price to zero in order to confuse other competitors.
    """

    def __init__(self):
        self.period = 0
        self._price = self._explore()
        self.prices = []
        self.quantities = []

    def _set_price(self, period):
        if period <= 40:
            return self._explore()
        else:
            rnd = random.uniform(0, 1)
            if rnd < 0.01:
                return self._disrupt_competitors()
            elif rnd < 0.05:
                return self._explore()
            else:
                return self._exploit()

    def _explore(self):
        return random.uniform(0, 100)

    def _exploit(self):

        model = train_linear_regression(self.prices, self.quantities)

        prices = list(range(0, 100))
        predicted_q = predict_quantity(model, prices)
        revenues = [q * p for q, p in zip(prices, predicted_q)]
        idx_max = revenues.index(max(revenues))
        return prices[idx_max]

    def _disrupt_competitors(self):
        return 0

    @property
    def price(self):
        return self._price

    def observe_market(self, prices, q):
        self.period += 1
        self.prices.append(prices[self])
        self.quantities.append(q)
        self._price = self._set_price(self.period)


def train_linear_regression(
    prices: List[float], quantities: List[float]
) -> sklearn.linear_model.LinearRegression:
    model = sklearn.linear_model.LinearRegression()
    model.fit(np.array(prices).reshape(-1, 1), quantities)
    return model


def predict_quantity(model, prices: List[float]) -> List[float]:
    price_array = np.array(prices).reshape(-1, 1)
    quantity_array = model.predict(price_array)
    return list(quantity_array)


class GreedyFirm:
    def __init__(self):
        self.all_prices = [[]]
        self._price = random.uniform(0, 100)

    def observe_market(self, prices, q):
        last_prices = [price for firm, price in prices.items()]
        self._price = self._set_price()
        self.all_prices.append(last_prices)

    def _set_price(self):
        last_period_prices = self.all_prices[-1]

        if last_period_prices:
            min_price = min(last_period_prices)
        else:
            min_price = 0

        prices_last_30_periods = [
            p for period_prices in self.all_prices[:-30] for p in period_prices
        ]

        if prices_last_30_periods:
            lower_10 = np.percentile(prices_last_30_periods, 10)
            print(lower_10)
        else:
            lower_10 = 0

        if min_price < lower_10:
            return min(lower_10, 5)
        else:
            return min_price

    @property
    def price(self):
        return self._price


if __name__ == '__main__':

    def q(p):
        quantity = 1000 - 10 * p + random.gauss(0, 20)
        return quantity if quantity > 0 else 0

    firm = GreedyFirm()

    for _ in range(1000):
        prices = {
            random.uniform(0, 100): round(random.uniform(10, 100), 2)
            for _ in range(100)
        }
        firm.observe_market(prices=prices, q=q(firm.price))
        if _ % 10 == 0:
            print(firm.price)
