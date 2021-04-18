import numpy as np
from dynpric.types import Firm
from dynpric.types import PricesSet


class PoissonDemand:
    """
    Poisson demand for a single firm.
    """

    def __init__(self) -> None:
        ...

    @staticmethod
    def allocate(prices_set: PricesSet) -> dict[Firm, int]:
        ((firm, price),) = prices_set.items()

        def λ(p: float) -> float:
            return max(100 - 2 * p, 0)

        q: int = np.random.poisson(λ(price))
        return {firm: q}
