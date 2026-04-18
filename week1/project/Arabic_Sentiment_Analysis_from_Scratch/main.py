import random
from datasets import load_dataset

from arabic_sentiment.preprocessing import ArabicPreprocessor
from arabic_sentiment.language_model import NgramLanguageModel
from arabic_sentiment.naive_bayes import NaiveBayesClassifier
from arabic_sentiment.evaluation import accuracy, precision_recall_f1, confusion_matrix_str


ds = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus")
train_data = ds["train"]
test_data  = ds["test"]

# ----------------------------------- Pre Processing -----------------------------------------
pre_processor = ArabicPreprocessor()

train_data = train_data.map(lambda x: {"clean_tweet": pre_processor.preprocess(x["tweet"])})
test_data = test_data.map(lambda x: {"clean_tweet": pre_processor.preprocess(x["tweet"])})

raw_train = [x.split() for x in train_data["tweet"]]
clean_train = train_data["clean_tweet"]
raw_test = [x.split() for x in test_data["tweet"]]
clean_test = test_data["clean_tweet"]

# ------------------------------------- Language Model -------------------------------------------

models = {
    "Bigram":   NgramLanguageModel(2),
    "Bigram (Clean)": NgramLanguageModel(2),
    "Trigram":  NgramLanguageModel(3),
    "Trigram (Clean)": NgramLanguageModel(3)
}

for name, model in models.items():
    if "Clean" in name:
        model.train(clean_train)
    else:
        model.train(raw_train)


results = {}
for name, model in models.items():
    if "Clean" in name:
        results[name] = model.perplexity(clean_test)
    else:
        results[name] = model.perplexity(raw_test)


print("\n" + "-"*45)
print(f" {'Model'}  | {'Raw PPL'} | {'Preprocessed'}")
print("-" * 45)
print(f"{'Bigram'}  | {results['Bigram']} | {results['Bigram (Clean)']}")
print(f"{'Trigram'} | {results['Trigram']} | {results['Trigram (Clean)']}")
print("-"*45 + "\n")


for name, model in models.items():
    print(f"--- Generated Samples from [{name}] ---")
    for i in range(3):
        sample = model.generate()
        print(f"{i+1}. {sample}")
    print("-" * 30)

# ------------------------------------------- Naive Bayes Classifier -----------------------------------------------
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