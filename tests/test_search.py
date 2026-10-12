import numpy as np
from rxdx.core import Passage
from rxdx.search import SemanticIndex

class FakeModel:
    def encode(self,texts,normalize_embeddings):
        assert normalize_embeddings
        return np.array([[1.,0.] if 'clinic' in t else [0.,1.] for t in texts])

def test_cosine_ranking_and_threshold():
    passages=[Passage(1,'https://rxdx.in/','clinic location'),Passage(2,'https://rxdx.in/','vaccination services')]
    index=SemanticIndex(passages,FakeModel())
    assert index.search('clinic address')==[passages[0]]
    assert index.search('vaccination')==[passages[1]]
