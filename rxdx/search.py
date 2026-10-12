import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticIndex:
    def __init__(self, passages, model=None):
        if not passages:
            raise ValueError('No passages to index.')
        self.passages = passages
        self.model = model or SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.vectors = self.model.encode([p.text for p in passages], normalize_embeddings=True)

    def search(self, question, limit=4, threshold=0.3):
        query = self.model.encode([question], normalize_embeddings=True)[0]
        scores = np.asarray(self.vectors) @ query
        order = np.argsort(-scores)[:limit]
        return [self.passages[i] for i in order if scores[i] >= threshold]
