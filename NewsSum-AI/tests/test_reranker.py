from newsum.inference.reranker import rerank, canonicalize_entity

class Entity:
    def __init__(self, text):
        self.text = text
        self.label_ = 'PERSON'

class Doc:
    def __init__(self, text):
        self.ents = [Entity('Alice')] if 'Alice' in text else []

class DummyNLP:
    def __call__(self, text):
        return Doc(text)

class DummyNLI:
    def score(self, source, candidate):
        return .9 if 'Alice' in candidate else .2


def test_reranker_picks_faithful():
    selected, scored = rerank('Alice arrived', [('Alice arrived', -.7),('Nobody arrived', -.2)],
                              DummyNLI(), DummyNLP())
    assert selected.summary == 'Alice arrived'
    assert len(scored) == 2


def test_entity_norm():
    assert canonicalize_entity('The Example, Inc.') == 'example inc'
