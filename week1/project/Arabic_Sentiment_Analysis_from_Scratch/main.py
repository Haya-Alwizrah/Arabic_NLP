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
naive = NaiveBayesClassifier()

naive.train(clean_train, train_data["label"])

# Take 100 Sample:
random.seed(42)
random_test_data = random.sample(range(len(test_data)), 100)
sample_test = test_data.select(random_test_data)

sample_test_row = sample_test["tweet"]
sample_test_clean = sample_test["clean_tweet"]
sample_test_label = sample_test["label"]

# Predect :
predictions = naive.predict(sample_test_clean)

# ------------------------------------- evaluation --------------------------------------
acc = accuracy(predictions, sample_test_label)
prec, rec, f1 = precision_recall_f1(predictions, sample_test_label, positive_label=1)

print("--- CLASSIFICATION REPORT ---")
print(f"Accuracy  : {acc}")
print(f"Precision : {prec}")
print(f"Recall    : {rec}")
print(f"F1-Score  : {f1}")

print("\n--- CONFUSION MATRIX ---")
cm = confusion_matrix_str(predictions, sample_test_label, labels=[1, 0])
print(cm)

# ------------------------------------------------------------------------------

correct_examples = []
incorrect_examples = []

for i in range(len(predictions)):
    item = {
        "tweet": sample_test_row[i],
        "true": sample_test_label[i],
        "pred": predictions[i]
    }
    if predictions[i] == sample_test_label[i]:
        correct_examples.append(item)
    else:
        incorrect_examples.append(item)

print("\n" + "="*30 + " 5 CORRECT PREDICTIONS " + "="*30)
for i, ex in enumerate(correct_examples[:5]):
    print(f"{i+1}. [True: {ex['true']} | Pred: {ex['pred']}] -> {ex['tweet']}")

print("\n" + "="*30 + " 5 INCORRECT PREDICTIONS " + "="*30)
for i, ex in enumerate(incorrect_examples[:5]):
    print(f"{i+1}. [True: {ex['true']} | Pred: {ex['pred']}] -> {ex['tweet']}")