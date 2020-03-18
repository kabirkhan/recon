from typing import Dict


class Scorer:
    def __init__(self) -> None:
        self._scores: Dict[str, object] = {}

    @property
    def scores(self) -> Dict[str, object]:
        return self._scores
