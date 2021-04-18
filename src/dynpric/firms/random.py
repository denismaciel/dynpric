from dynpric.types import History
from numpy.random import uniform


class RandomFirm:
    def __init__(self, name: str, min: int, max: int):
        self.name = name
        self.min = min
        self.max = max
        self._set_price()

    def observe_market(self, history: History) -> None:
        prices_set, demand = history[-1]
        assert self.price == prices_set[self]
        self._set_price()

    def _set_price(self) -> None:
        self._price = round(uniform(self.min, self.max))

    @property
    def price(self) -> float:
        return self._price

    def __repr__(self) -> str:
        return f'RandomFirm(name={self.name})'
