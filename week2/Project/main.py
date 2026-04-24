from datasets import load_dataset
from preprocessing import ArabicPreprocessor
from week2.Project.Arab2v import Arab_w2v
import numpy as np

ds = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus")

# --------------------------------------[ Part1 ]---------------------------------------
# Pre Processing:

pre_processor = ArabicPreprocessor()
data = ds.map(lambda x: {"clean_tweet": pre_processor.preprocess(x["tweet"])})
train_data = data["train"].remove_columns(["tweet"])
test_data  = data["test"].remove_columns(["tweet"])

# w2v:
w2v = Arab_w2v(100,5,1,0)
w2v.train(train_data["clean_tweet"])
print(w2v.most_similar("رمضان"))
print(w2v.similarity("الهلال", "النصر"))
w2v.save("w2v.model")

# --------------------------------------[ Part 2 ]---------------------------------------
