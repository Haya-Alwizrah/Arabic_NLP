from gensim.models import Word2Vec

class Arab_w2v():
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
        return self.model.wv.most_similar(word)   
    
    def similarity(self, word1, word2):
        return self.model.wv.similarity(word1, word2)

    def save(self, path="word2vec.model"):
        self.model.save(path)