from datasets import load_dataset
from preprocessing import ArabicPreprocessor
from Arab2v import Arab2v
import numpy as np

ds = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus")

# --------------------------------------[ Part1 ]---------------------------------------
# Pre Processing:

pre_processor = ArabicPreprocessor()
data = ds.map(lambda x: {"clean_tweet": pre_processor.preprocess(x["tweet"])})
train_data = data["train"].remove_columns(["tweet"])
test_data  = data["test"].remove_columns(["tweet"])

# w2v:
a2v = Arab2v(100,5,1,0)
a2v.train(train_data["clean_tweet"])
print(a2v.most_similar("رمضان"))
print(a2v.similarity("الهلال", "النصر"))
a2v.save("week2\Project\w2v.model")

# --------------------------------------[ Part 2 ]---------------------------------------
X_train = a2v.s2v(train_data["clean_tweet"])
X_test  = a2v.s2v(test_data["clean_tweet"])
y_train = train_data["label"]
y_test = test_data["label"]