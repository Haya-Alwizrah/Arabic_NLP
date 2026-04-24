from gensim.models import Word2Vec
import numpy as np

class Arab2v():
    def __init__(self, vec_size, win, min_c,sg):
        self.vec_size = vec_size
        self.win = win
        self.min_c = min_c
        self.sg = sg
        
    def train(self, data):
        self.model = Word2Vec(
            sentences = data,
            vector_size = self.vec_size,
            window = self.win,
            min_count = self.min_c,
            sg = self.sg
        )

    def most_similar(self, word):
        return self.model.wv.most_similar(word)[0]
    
    def similarity(self, word1, word2):
        return float(self.model.wv.similarity(word1, word2))

    def save(self, path="word2vec.model"):
        self.model.save(path)

    def s2v(self, data):
        vs = self.model.vector_size
        s2v = []

        for sent in data:
            vec = []

            for w in sent:
                if w in self.model.wv:
                    vec.append(self.model.wv[w])

            if len(vec) == 0:
                sent_vec = np.zeros(vs)
            else:
                sent_vec = np.mean(vec, axis=0)

            s2v.append(sent_vec)

        return np.array(s2v)    