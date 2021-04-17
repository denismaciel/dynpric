from abc import ABC

from dynpric.priors import Prior


class Seller(ABC):
    def __init__(
        self,
        beliefs: list[Prior],
    ) -> None:
        self.beliefs = beliefs

    def update_beliefs(self) -> None:
        ...
