

class Scorer:
    def __init__(self):
        self._scores = {}
    
    @property
    def scores(self):
        return self._scores