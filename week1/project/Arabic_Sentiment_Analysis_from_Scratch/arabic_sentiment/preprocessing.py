import re
from typing import List

class ArabicPreprocessor:
    """
    A preprocessing pipeline for Arabic Twitter text.
    
    Each method is a standalone transformation step.
    The `preprocess` method chains them together.
    """

    def remove_diacritics(self, text: str) -> str:
        """
        Remove Arabic diacritics (tashkeel/harakat).
        
        Arabic diacritics are Unicode characters in the range U+064B to U+065F.
        They represent short vowels and are usually absent in informal text,
        so we normalize by removing them.
        
        Hint: use re.sub with a Unicode range pattern.
        """

        text = re.sub("[\u064B-\u065F]","",text)
        return text

    def normalize_alef(self, text: str) -> str:
        """
        Normalize all Alef variants (أ إ آ ٱ) to plain Alef (ا).
        
        This reduces vocabulary sparsity because the same word
        may be written with different Alef forms on social media.
        """

        text = re.sub("[أإآٱ]","ا",text)
        return text

    def normalize_teh_marbuta(self, text: str) -> str:
        """
        Normalize Teh Marbuta (ة) to Heh (ه).
        
        Some writers use these interchangeably, especially in informal text.
        """

        text = re.sub("ة","ه",text)
        return text

    def remove_urls(self, text: str) -> str:
        """Remove HTTP/HTTPS URLs from text."""

        text = re.sub(r"http\S+","",text)
        return text

    def remove_mentions(self, text: str) -> str:
        """Remove Twitter @mentions."""

        text = re.sub(r"@\S+","",text)
        return text

    def remove_hashtags(self, text: str) -> str:
        """
        Remove the '#' symbol but KEEP the word.
        
        Hashtags often carry sentiment signal, so we keep the word
        while removing the special character.
        """
        text = re.sub("#","",text)
        return text

    def remove_punctuation_and_emojis(self, text: str) -> str:
        """
        Remove punctuation and emoji characters.
        
        Hint: Unicode ranges for emojis include U+1F300–U+1F9FF and others.
        """
        text = re.sub("[^ا-ي0-9]+"," ",text).strip()
        return text

    def remove_repeated_characters(self, text: str) -> str:
        """
        Normalize elongated words, e.g., 'جميييييل' → 'جميل'.
        
        Arabic writers often repeat characters for emphasis.
        Collapse any character repeated more than twice to a single character.
        """
        text = re.sub(r"(.)\1{2,}", r"\1\1", text) 
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text by splitting on whitespace.
        
        After cleaning, a simple whitespace split is sufficient.
        Filter out any empty strings.
        
        Returns:
            A list of word tokens.
        """
        lst = text.split()
        lst2 = []
        for l in lst:
            if l.strip():
                lst2.append(l)

        return lst2


    def preprocess(self, text: str, tokenize: bool = True):
        """
        Run the full preprocessing pipeline.
        
        Apply all steps in a sensible order, then optionally tokenize.
        
        Args:
            text:      Raw input string.
            tokenize:  If True, return List[str]; otherwise return cleaned str.
        
        Returns:
            List of tokens or a cleaned string.
        """
        text1 = self.remove_diacritics(text)
        text2 = self.normalize_alef(text1)
        text3 = self.normalize_teh_marbuta(text2)
        text4 = self.remove_repeated_characters(text3)
        text5 = self.remove_urls(text4)
        text6 = self.remove_mentions(text5)
        text7 = self.remove_hashtags(text6)
        text8 = self.remove_punctuation_and_emojis(text7)
        
        if tokenize:
            lst = self.tokenize(text8)
            return(lst)
        else:
            return(text8)
        

