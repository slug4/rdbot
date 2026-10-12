"""Bundled, checksum-verified local sentence embeddings; no remote AI calls."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).with_name('models')


class LocalEmbeddings:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        for entry in json.loads((ROOT/'manifest.json').read_text()):
            digest=hashlib.sha256((ROOT/entry['file']).read_bytes()).hexdigest()
            if digest!=entry['sha256']:
                raise RuntimeError('Bundled model integrity check failed')
        options=ort.SessionOptions()
        options.intra_op_num_threads=1
        options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(ROOT/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.tokenizer=Tokenizer.from_file(str(ROOT/'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0,pad_token='[PAD]')

    def encode(self,texts):
        results=[]
        for start in range(0,len(texts),4):
            tokens=self.tokenizer.encode_batch(texts[start:start+4])
            values={'input_ids':np.asarray([t.ids for t in tokens],dtype=np.int64),
                    'attention_mask':np.asarray([t.attention_mask for t in tokens],dtype=np.int64),
                    'token_type_ids':np.asarray([t.type_ids for t in tokens],dtype=np.int64)}
            inputs={item.name:values[item.name] for item in self.session.get_inputs()}
            embeddings=self.session.run(None,inputs)[0]
            mask=values['attention_mask'][...,None]
            vectors=(embeddings*mask).sum(axis=1)/np.maximum(mask.sum(axis=1),1)
            vectors=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
            results.extend(vectors)
        return np.asarray(results,dtype=np.float32)


@lru_cache(maxsize=1)
def get_model():
    return LocalEmbeddings()
