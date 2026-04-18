from datasets import load_dataset

ds = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus")

train_data = ds["train"]
test_data  = ds["test"]

import re
from arabic_sentiment.preprocessing import ArabicPreprocessor

p = ArabicPreprocessor()

ds2 = []
for d in train_data["tweet"]:
    ds2.append(p.preprocess(d))

for i in zip(train_data["tweet"][:5],ds2[:5]):
    print(i)

