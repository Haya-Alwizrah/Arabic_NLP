import numpy as np

class SentenceRepresentation:
    def __init__(self, model):
        self.model = model

    def s2v(self, sent):
        vec = []

        for w in sent:
            if w in self.model.wv:
                vec.append(self.model.wv[w])

        if len(vec) == 0:
            return np.zeros(self.vector_size)

        return np.mean(vec, axis=0)

    def transform(self, dataset):
        return np.array([
            self.sentence_to_vec(sent)
            for sent in dataset
        ])