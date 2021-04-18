import random

import numpy as np
import sklearn.linear_model
from dynpric.types import History
from dynpric.types import Price


def train_linear_regression(
    prices: list[float], quantities: list[float]
) -> sklearn.linear_model.LinearRegression:
    model = sklearn.linear_model.LinearRegression()
    model.fit(np.array(prices).reshape(-1, 1), quantities)
    return model


def predict_quantity(
    model: sklearn.linear_model.LinearRegression, prices: list[int]
) -> list[float]:
    price_array = np.array(prices).reshape(-1, 1)
    quantity_array = model.predict(price_array)
    return list(quantity_array)


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

    def __init__(self, name: str) -> None:
        self.name = name
        self._price = self._explore()

    def _set_price(self, period: int) -> Price:
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

    @staticmethod
    def _explore() -> Price:
        return random.uniform(0, 100)

    def _exploit(self) -> Price:

        model = train_linear_regression(self.prices, self.quantities)

        prices = list(range(0, 100))
        predicted_q = predict_quantity(model, prices)
        revenues = [q * p for q, p in zip(prices, predicted_q)]
        idx_max = revenues.index(max(revenues))
        return prices[idx_max]

    def _disrupt_competitors(self) -> float:
        return 0.0

    def observe_market(self, history: History) -> None:
        self.prices = [prices[self] for prices, _ in history]
        self.quantities = [demand[self] for _, demand in history]
        period = len(history)
        self._price = self._set_price(period)

    @property
    def price(self) -> Price:
        return self._price

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.name})'


class GreedyFirm:
    """
    TODO
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._price = random.uniform(0, 100)

    @staticmethod
    def _set_prices(all_prices: list[Price], last_period_prices: list[Price]) -> Price:

        min_price = min(last_period_prices) if last_period_prices else 0
        lower_10 = np.percentile(all_prices, 10) if all_prices else 0
        price = max(lower_10, 5) if min_price < lower_10 else min_price
        return price

    def observe_market(self, history: History) -> None:
        all_prices = [p for prices, _ in history[-30:] for _, p in prices.items()]

        prices, _ = history[-1]
        last_period_prices = [p for _, p in prices.items()]

        self._price = self._set_prices(all_prices, last_period_prices)

    @property
    def price(self) -> Price:
        return self._price

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.name})'
