from ..scoring import Ranker, Tokens as PairTokens, pack, trim

INSTRUCTION = 'Retrieve a legal passage that answers the query, respecting its legal issues and jurisdiction.'


class Tokens(PairTokens):
    def __init__(self, path, cache, instruction=INSTRUCTION):
        super().__init__(path, cache, instruction)
