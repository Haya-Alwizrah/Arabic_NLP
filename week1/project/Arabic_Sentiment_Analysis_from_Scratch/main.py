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

from arabic_sentiment.naive_bayes import NaiveBayesClassifier
from arabic_sentiment.evaluation import accuracy, precision_recall_f1, confusion_matrix_str
import random

naiv = NaiveBayesClassifier()
naiv.train(clean_train, train_data["label"])

random.seed(42)
test_indices = random.sample(range(len(test_data)), 100)
sample_test = test_data.select(test_indices)

sample_texts = [x.split() for x in sample_test["clean_tweet"]]
gold_labels = sample_test["label"]
original_tweets = sample_test["tweet"]

predictions = naiv.predict(sample_texts)

acc = accuracy(predictions, gold_labels)
prec, rec, f1 = precision_recall_f1(predictions, gold_labels, positive_label="pos")

print("--- CLASSIFICATION REPORT ---")
print(f"Accuracy  : {acc}")
print(f"Precision : {prec}")
print(f"Recall    : {rec}")
print(f"F1-Score  : {f1}")

print("\n--- CONFUSION MATRIX ---")
cm_str = confusion_matrix_str(predictions, gold_labels, labels=["pos", "neg"])
print(cm_str)

correct_examples = []
incorrect_examples = []

for i in range(len(predictions)):
    item = {
        "tweet": original_tweets[i],
        "true": gold_labels[i],
        "pred": predictions[i]
    }
    if predictions[i] == gold_labels[i]:
        correct_examples.append(item)
    else:
        incorrect_examples.append(item)

print("\n" + "="*30 + " 5 CORRECT PREDICTIONS " + "="*30)
for i, ex in enumerate(correct_examples[:5]):
    print(f"{i+1}. [True: {ex['true']} | Pred: {ex['pred']}] -> {ex['tweet']}")

print("\n" + "="*30 + " 5 INCORRECT PREDICTIONS " + "="*30)
for i, ex in enumerate(incorrect_examples[:5]):
    print(f"{i+1}. [True: {ex['true']} | Pred: {ex['pred']}] -> {ex['tweet']}")