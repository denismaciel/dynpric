import random

import numpy as np
from dynpric.types import Firm
from dynpric.types import Price
from dynpric.types import PriceLevel
from dynpric.types import PricesSet
from dynpric.types import Quantity


class PoissonDemand:
    """
    Poisson demand for a single firm.
    """

    def __init__(self) -> None:
        ...

    def allocate(self, prices_set: PricesSet) -> dict[Firm, int]:
        try:
            ((firm, price),) = prices_set.items()
        except ValueError:
            msg = f'{type(self).__name__} only supports one firm. {len(prices_set)} firms passed.'
            raise RuntimeError(msg)

        def λ(p: float) -> float:
            return max(100 - 2 * p, 0)

        q: int = np.random.poisson(λ(price))
        return {firm: q}


class BernoulliDemand:
    def __init__(self, price_levels: list[PriceLevel]) -> None:
        self.price_levels = price_levels

    @staticmethod
    def _simulate_buying_decision(price_level: PriceLevel) -> Quantity:
        if random.random() > price_level.true_prob:
            return 0
        return 1

    def allocate(self, prices_set: PricesSet) -> dict[Firm, int]:
        try:
            ((firm, price),) = prices_set.items()
        except ValueError:
            msg = f'{type(self).__name__} only supports one firm. {len(prices_set)} firms passed.'
            raise RuntimeError(msg)

        if price == np.inf:
            demand = 0
        else:
            try:
                (price_level,) = [pl for pl in self.price_levels if pl.price == price]
            except ValueError:
                raise ValueError(f'Price {price} is not an allowed price.')
            demand = self._simulate_buying_decision(price_level)
        return {firm: demand}

    def realize_demand(self, p: Price) -> Quantity:
        for pl in self.price_levels:
            if pl.price == p:
                return self._simulate_buying_decision(pl)
        else:
            raise ValueError(f'Price {p} is not an allowed price.')
