from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import NamedTuple

import numpy as np


Price = float


class Prior(ABC):
    """
    Abstract base class for priors
    """

    @property
    @abstractmethod
    def params(self) -> Dict[str, float]:
        ...

    @property
    @abstractmethod
    def expected_value(self) -> float:
        ...

    @abstractmethod
    def sample(self, n: int) -> float:
        """
        Samples n values from distribution
        """

    @abstractmethod
    def update(self, result: int) -> None:
        """
        Updates parameters with new information (result)
        """


class BetaPrior(Prior):
    def __init__(self, alpha: int, beta: int) -> None:
        self.alpha = alpha
        self.beta = beta

    @property
    def params(self) -> Dict[str, float]:
        return {'alpha': self.alpha, 'beta': self.beta}

    @property
    def expected_value(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self, n: int = 1) -> float:
        return np.random.beta(self.alpha, self.beta, size=n)  # type: ignore

    def update(self, result: int) -> None:

        if result not in (0, 1):
            raise ValueError('`result` must be either 0 or 1')

        self.alpha += result
        self.beta += 1 - result

    def __repr__(self) -> str:
        return f'BetaPrior(alpha={self.alpha}, beta={self.beta})'


class Belief(NamedTuple):
    price: Price
    prior: Prior
