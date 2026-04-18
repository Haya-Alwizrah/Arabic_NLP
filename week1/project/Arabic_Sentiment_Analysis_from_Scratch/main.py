from datasets import load_dataset

ds = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus")

train_data = ds["train"]
test_data  = ds["test"]

import re
from arabic_sentiment.preprocessing import ArabicPreprocessor
from arabic_sentiment.language_model import NgramLanguageModel


p = ArabicPreprocessor()

train_data = train_data.map(lambda x: {"clean_tweet": p.preprocess(x["tweet"])})

model_1 = NgramLanguageModel(2)
model_2 = NgramLanguageModel(2)
model_3 = NgramLanguageModel(3)
model_4 = NgramLanguageModel(3)

raw_train = [x.split() for x in train_data["tweet"]]
clean_train = train_data["clean_tweet"]

model_1.train(raw_train)
model_2.train(clean_train)
model_3.train(raw_train)
model_4.train(clean_train)


test_data = test_data.map(lambda x: {"clean_tweet": p.preprocess(x["tweet"])})
raw_test = [x.split() for x in test_data["tweet"]]
clean_test = test_data["clean_tweet"]

p_raw1 = model_1.perplexity(raw_test)
p_clean1 = model_2.perplexity(clean_test)
p_raw2 = model_3.perplexity(raw_test)
p_clean2 = model_4.perplexity(clean_test)

print("\n" + "="*45)
print(f"{'Model':<15} | {'Raw PPL':<12} | {'Preprocessed':<12}")
print("-" * 45)
print(f"{'Bigram':<15} | {p_raw1:<12.2f} | {p_clean1:<12.2f}")
print(f"{'Trigram':<15} | {p_raw2:<12.2f} | {p_clean2:<12.2f}")
print("="*45 + "\n")

models = {
    "Bigram (Raw)": model_1,
    "Bigram (Clean)": model_2,
    "Trigram (Raw)": model_3,
    "Trigram (Clean)": model_4
}

for name, model in models.items():
    print(f"--- Generated Samples from [{name}] ---")
    for i in range(3):
        sample = model.generate()
        print(f"{i+1}. {sample}")
    print("-" * 30)