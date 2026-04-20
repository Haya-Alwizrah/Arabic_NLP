import gensim.downloader
from sklearn.metrics.pairwise import cosine_similarity
from numpy import argmax

class W2V():
    def __init__(self, model_name):
        self.model = gensim.downloader.load(model_name)
        self.data = []
        self.data_embd = []
    
    def embding(self,text):
        words = text.split()
        vectors = []

        for word in words:
            vector = self.model[word]
            vectors.append(vector)
        final = sum(vectors) / len(vectors)
        return final.tolist()
    

    def add_data(self, data):
        self.data = data
        for d in data:
            self.data_embd.append(self.embding(d))

    def searcher(self, query):
        q_embd = self.embding(query)

        cos = cosine_similarity([q_embd],self.data_embd)

        return self.data[argmax(cos).item()]
    
    def most_similar(self, word):
        output = self.model.most_similar(word)[0]
        return (output[0], round(output[1] * 100, 2))

# ---------------------------------------------- Main --------------------------------------------

w2v = W2V('glove-twitter-25')

['fasttext-wiki-news-subwords-300',
 'conceptnet-numberbatch-17-06-300',
 'word2vec-ruscorpora-300',
 'word2vec-google-news-300',
 'glove-wiki-gigaword-50',
 'glove-wiki-gigaword-100',
 'glove-wiki-gigaword-200',
 'glove-wiki-gigaword-300',
 'glove-twitter-25',
 'glove-twitter-50',
 'glove-twitter-100',
 'glove-twitter-200',
 '__testing_word2vec-matrix-synopsis']

print(w2v.most_similar("twitter"))

database = [
    "serry was here yesterday",
    "obama did not die",
    "mark is a lizard",
    "girls no answer",
    "twitter is bad"
]
w2v.add_data(database)

c = w2v.searcher("when did obama died")
print(c)
