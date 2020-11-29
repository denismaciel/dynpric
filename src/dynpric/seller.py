from abc import ABC
from typing import List

from dynpric.priors import Prior


class Seller(ABC):
    def __init__(
        self,
        beliefs: List[Prior],
    ) -> None:
        self.beliefs = beliefs

    def update_beliefs(self) -> None:
        ...
