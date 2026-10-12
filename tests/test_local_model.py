import numpy as np
import pytest
from local_model import get_model, LocalEmbeddings
from rag import Chunk, Retriever


def test_real_local_model_retrieves_semantically_without_network(monkeypatch):
    def fail(*_,**__): raise AssertionError('Local model must not use network')
    monkeypatch.setattr('requests.get',fail)
    model=get_model()
    chunks=[Chunk('The hospital is located on Example Road.','https://rxdx.in/'),Chunk('Vaccination appointments are available.','https://rxdx.in/')]
    index=Retriever(chunks,model)
    hits=index.search('Where is the hospital?')
    assert hits[0][0]==chunks[0]
    assert model.encode(['hospital location']).shape==(1,384)
    assert np.isclose(np.linalg.norm(model.encode(['hospital location'])[0]),1,atol=1e-5)


def test_model_checksum_mismatch_is_rejected(tmp_path,monkeypatch):
    import json
    (tmp_path/'manifest.json').write_text(json.dumps([{'file':'model.onnx','sha256':'wrong'}]))
    (tmp_path/'model.onnx').write_bytes(b'untrusted')
    monkeypatch.setattr('local_model.ROOT',tmp_path)
    with pytest.raises(RuntimeError,match='integrity'):
        LocalEmbeddings()
