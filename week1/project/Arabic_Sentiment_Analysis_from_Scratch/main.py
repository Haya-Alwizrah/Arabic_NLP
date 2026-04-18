from datasets import load_dataset

ds = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus")

train_data = ds["train"]
test_data  = ds["test"]