import numpy as np
from dynpric.types import Prior


class BetaPrior(Prior):
    def __init__(self, α: int, β: int) -> None:
        self.α = α
        self.β = β

    @property
    def params(self) -> dict[str, float]:
        return {'α': self.α, 'β': self.β}

    @property
    def expected_value(self) -> float:
        return self.α / (self.α + self.β)

    def sample(self) -> float:
        return np.random.beta(self.α, self.β)  # type: ignore

    def update(self, result: int) -> None:

        if result not in (0, 1):
            raise ValueError('`result` must be either 0 or 1')

        self.α += result
        self.β += 1 - result

    def __repr__(self) -> str:
        return f'{type(self).__name__}(α={self.α}, β={self.β})'


class GammaPrior(Prior):
    def __init__(self, α: int, β: int) -> None:
        self.α = α
        self.β = β

    @property
    def expected_value(self) -> float:
        return self.α / self.β

    def sample(self) -> float:
        """A Gamma distribution can be parameterized in two different, but
        equivalent ways.  We can either refer to α and β or shape and scale
        numpy uses the latter np.random.gamma(shape, scale).

        The two are related as follows:
          * α = scale
          * 1/β = shape
        """
        return int(np.random.gamma(self.α, 1 / self.β))

    def update(self, result: int) -> None:
        self.α += result
        self.β += 1

    @property
    def params(self) -> dict[str, float]:
        return {'α': self.α, 'β': self.β}

    def __repr__(self) -> str:
        return f'{type(self).__name__}(α={self.α}, β={self.β})'
