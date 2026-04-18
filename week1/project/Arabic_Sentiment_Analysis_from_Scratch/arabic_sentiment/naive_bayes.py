from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import math

Label = str   # "positive" or "negative"

class NaiveBayesClassifier:
    """
    Multinomial Naïve Bayes classifier for text.

    Supports Laplace (add-k) smoothing. Operates on pre-tokenized input.
    
    Attributes:
        k:               Smoothing parameter (default 1.0).
        class_log_priors: log P(c) for each class.
        word_log_likelihoods: log P(w | c) for each class and word.
        vocab:           All words seen during training.
    """

    def __init__(self, k: float = 1.0):
        self.k = k
        self.log_priors = {}
        self.log_likelihoods = {}
        self.vocab = set()

    def train(self, documents: List[List[str]], labels: List[Label]) -> None:
        """
        Estimate log priors and log likelihoods from training data.
        
        Steps:
            1. Count documents per class  → compute log priors.
            2. Concatenate all tokens per class → build per-class word counts.
            3. Apply Laplace smoothing → compute log likelihoods.
        
        Store smoothed values as log probabilities to avoid underflow.
        
        Args:
            documents: List of tokenized documents.
            labels:    Corresponding class label for each document.
        """
        n_doc = len(documents)
        for doc in documents:
            for word in doc:
                self.vocab.add(word)

        # log priors:
        c_label = {}
        for l in labels:
            if l in c_label:
                c_label[l] += 1
            else:
                c_label[l] = 1

        c_word = {}
        t_words = {}

        for doc , label in zip(documents, labels):
            if label not in c_word:
                c_word[label] = {}
                t_words[label] = 0
            for word in doc:
                if word in c_word[label]:
                    c_word[label][word] += 1
                else:
                    c_word[label][word] = 1
                
                t_words[label] += 1
        
        for l in c_label: # c_label: exa : {("pos", 3),("neg", 4)}
            self.log_priors[l] = math.log2(c_label[l] / n_doc) # عدد اللي متصنفين بذا القسم على عدد الكل
            
            self.log_likelihoods[l] = {}
            for word in self.vocab:
                if word in c_word[l]:
                    count_w_l = c_word[l][word]
                else:
                    count_w_l = 0
                
                prob = (count_w_l + 1) / (t_words[l] + len(self.vocab))
                self.log_likelihoods[l][word] = math.log2(prob)


    def predict_one(self, tokens: List[str]) -> Label:
        """
        Predict the class of a single tokenized document.
        
        For unknown words (not in vocab), skip them — do not crash.
        
        Returns:
            The predicted label.
        """
        labels = list(self.log_priors.keys())

        label = None
        score = -float('inf')

        for l in labels:
            c_score = self.log_priors[l]
            for w in tokens:
                if w in self.vocab:
                    c_score += self.log_likelihoods[l][w]
                else:
                    continue
            if c_score > score:
                score = c_score
                label = l

        return label

    def predict(self, documents: List[List[str]]) -> List[Label]:
        """
        Predict classes for a list of tokenized documents.
        
        Returns:
            A list of predicted labels, one per document.
        """
        lst = []
        for s in documents:
            s_label = self.predict_one(s)
            lst.append(s_label)
        return lst

    def top_features(self, n: int = 20) -> Dict[Label, List[Tuple[str, float]]]:
        """
        Return the top-n most discriminative words per class.
        
        Discriminative score for word w and class c:
            score(w, c) = log P(w | c) - log P(w | other_c)
        
        Returns:
            Dict mapping each label to a sorted list of (word, score) tuples.
        """
        top = {}
        labels = list(self.log_priors.keys())

        for label in labels:
            scores = []
            other_l = None
            for l in labels:
                if l != label:
                    other_l = l
                    break

            for w in self.vocab:
                log_c = self.log_likelihoods[label][w]
                log_other = self.log_likelihoods[other_l][w]
                score = log_c - log_other
                scores.append((w, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            top[label] = scores[:n]

        return top