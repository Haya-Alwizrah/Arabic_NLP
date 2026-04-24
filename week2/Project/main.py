from datasets import load_dataset
from preprocessing import ArabicPreprocessor



ds = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus")


# ----------------------------------- Pre Processing -----------------------------------------

pre_processor = ArabicPreprocessor()
data = ds.map(lambda x: {"clean_tweet": pre_processor.preprocess(x["tweet"])})
train_data = data["train"].remove_columns(["tweet"])
test_data  = data["test"].remove_columns(["tweet"])


