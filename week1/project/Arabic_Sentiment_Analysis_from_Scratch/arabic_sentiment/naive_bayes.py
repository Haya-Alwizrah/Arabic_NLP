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
        c_token = {}

        for doc , label in zip(documents, labels):
            if label not in c_word:
                c_word[label] = {}
                c_token[label] = 0
            for word in doc:
                if word in c_word[label]:
                    c_word[label][word] += 1
                else:
                    c_word[label][word] = 1
                
                c_token[label] += 1
        
        for l in c_label:
            self.log_priors[label] = math.log2(c_label[label] / n_doc)
            
            self.log_likelihoods[label] = {}
            for word in self.vocab:
                if word in c_word[label]:
                    count_w_l = c_word[label][word]
                else:
                    count_w_l = 0
                
                prob = (count_w_l + 1) / (c_token[label] + len(self.vocab))
                self.log_likelihoods[label][word] = math.log2(prob)


    def predict_one(self, tokens: List[str]) -> Label:
        """
        Predict the class of a single tokenized document.
        
        For unknown words (not in vocab), skip them — do not crash.
        
        Returns:
            The predicted label.
        """
        # TODO: implement
        raise NotImplementedError

    def predict(self, documents: List[List[str]]) -> List[Label]:
        """
        Predict classes for a list of tokenized documents.
        
        Returns:
            A list of predicted labels, one per document.
        """
        # TODO: implement using predict_one
        raise NotImplementedError

    def top_features(self, n: int = 20) -> Dict[Label, List[Tuple[str, float]]]:
        """
        Return the top-n most discriminative words per class.
        
        Discriminative score for word w and class c:
            score(w, c) = log P(w | c) - log P(w | other_c)
        
        Returns:
            Dict mapping each label to a sorted list of (word, score) tuples.
        """
        # TODO: implement
        raise NotImplementedError
