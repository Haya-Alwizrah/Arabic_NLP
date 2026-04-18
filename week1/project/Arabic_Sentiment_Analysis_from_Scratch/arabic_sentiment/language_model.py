from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Literal
import math

NGramOrder = Literal[2, 3]

class NgramLanguageModel:
    """
    A bigram or trigram language model with Laplace (add-1) smoothing.

    Attributes:
        n:       The order of the model (2 for bigram, 3 for trigram).
        vocab:   The set of all known tokens.
        counts:  Raw n-gram counts.
        context_counts: Counts of (n-1)-gram contexts.
    """

    def __init__(self, n: NGramOrder = 2):
        """
        Args:
            n: Order of the n-gram model. Must be 2 or 3.
        """
        self.n = n
        self.vocab = set()
        self.ngram_counts = {}
        self.context_counts = {}
  
    def extract_ngrams(self, tokens: List[str]) -> List[Tuple[str, ...]]: # تاخذ لست وتطلع لنا لست فيها الكلمه والكلمه اللي بعدها
        """
        Extract all n-grams from a token list.
        
        Remember to add special <s> start and </s> end tokens.
        For a trigram model, prepend TWO <s> tokens.
        
        Example (bigram): ['a', 'b', 'c'] → [('<s>','a'), ('a','b'), ('b','c'), ('c','</s>')]
        """
        lst = []
        if self.n == 2:
            tokens = ["<s>"] + tokens + ["</s>"]
            for l in range(len(tokens)-1):
                lst.append((tokens[l], tokens[l+1]))
        else:
            tokens = ["<s>", "<s>"] + tokens + ["</s>"]
            for l in range(len(tokens)-2):
                lst.append((tokens[l], tokens[l+1], tokens[l+2]))
        
        return lst

    def train(self, corpus: List[List[str]]) -> None: # تاخذ الداتا اللي نضفناها وهي لست اوف لست 
        """
        Train on a list of tokenized sentences.
        
        Args:
            corpus: A list of token lists (one per tweet/sentence).
        """
        
        for s in corpus:
            ngrams = self.extract_ngrams(s)

            for n in ngrams:
                # count n-grams
                if n in self.ngram_counts:
                    self.ngram_counts[n] += 1
                else:
                    self.ngram_counts[n] = 1
                
                # context counts
                c = n[:-1]
                if c in self.context_counts:
                    self.context_counts[c] += 1
                else:
                    self.context_counts[c] = 1

            #build vocab
            for w in s:
                self.vocab.add(w)

    def log_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Return the log probability of an n-gram using Laplace smoothing.
        
        Formula (Laplace):
            P(w | context) = (count(context, w) + 1) / (count(context) + |V|)
        
        Returns log base 2 of the probability.
        """
        # TODO: implement with smoothing
        raise NotImplementedError

    def sentence_log_probability(self, tokens: List[str]) -> float:
        """
        Return the total log probability of a tokenized sentence.
        
        This is the sum of log probabilities of each n-gram in the sentence.
        """
        # TODO: implement
        raise NotImplementedError

    def perplexity(self, corpus: List[List[str]]) -> float:
        """
        Compute perplexity on a held-out corpus.
        
        Perplexity = 2^(-average log probability per token)
        
        Lower perplexity = better model.
        """
        # TODO: implement
        raise NotImplementedError

    def generate(self, seed: List[str] = None, max_tokens: int = 20) -> str:
        """
        Generate a random sequence of tokens using the language model.
        
        Sample the next token proportionally to its probability given
        the current context, until </s> is generated or max_tokens is reached.
        
        Args:
            seed: Optional starting context (list of tokens).
        
        Returns:
            A generated string.
        """
        # TODO: implement (use random.choices with weights)
        raise NotImplementedError
