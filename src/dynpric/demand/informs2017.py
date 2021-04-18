import random
from typing import Collection
from typing import Sequence

import numpy as np
from dynpric.types import Allocation
from dynpric.types import Customer
from dynpric.types import Firm
from dynpric.types import PricesSet
from numpy.random import exponential
from numpy.random import multinomial
from numpy.random import poisson
from numpy.random import uniform


def assign_randomly(
    customers: Collection[Customer], firms: Sequence[Firm]
) -> Allocation:
    allocation: dict[Firm, list[Customer]] = {firm: [] for firm in firms}
    for c in customers:
        selected = random.choice(firms)
        allocation[selected].append(c)
    return allocation


def sample_shares(n: int) -> list[float]:
    samples: list[float] = [uniform() for _ in range(n)]
    total = sum(samples)
    shares = [x / total for x in samples]
    assert round(sum(shares), 5) == 1, sum(shares)
    return shares


class InformsDemand:
    def __init__(self) -> None:

        # Sample parameters
        self.λ = round(uniform(50, 150))
        self.θ_sho, self.θ_loy = sample_shares(2)

        self.β_sho: int = round(uniform(5, 15))
        self.β_loy: int = round(uniform(1.5, 2) * self.β_sho)

    def _sample_customers(self) -> tuple[np.ndarray, ...]:
        n = poisson(self.λ)
        n_sho, n_loy, = multinomial(
            n,
            (
                self.θ_sho,
                self.θ_loy,
            ),
        )
        assert n == n_sho + n_loy

        n_sho, n_loy,

        self.shoppers: np.ndarray = exponential(self.β_sho, n_sho)
        self.loyals: np.ndarray = exponential(self.β_loy, n_loy)

        return self.shoppers, self.loyals

    def allocate(self, prices_set: PricesSet) -> dict[Firm, int]:
        self._sample_customers()

        firms = list(prices_set.keys())
        assigned = assign_randomly(self.loyals, firms)

        return {
            firm: sum(c > firm.price for c in customers)
            for firm, customers in assigned.items()
        }
