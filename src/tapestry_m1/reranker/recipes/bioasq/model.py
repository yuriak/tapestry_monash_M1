from ..scoring import Ranker, Tokens as PairTokens, pack, trim

INSTRUCTION = 'Given a biomedical question, retrieve relevant biomedical abstracts that answer the question.'


class Tokens(PairTokens):
    def __init__(self, path, cache, instruction=INSTRUCTION):
        super().__init__(path, cache, instruction)
