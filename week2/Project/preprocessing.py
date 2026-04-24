import re
from typing import List

class ArabicPreprocessor:  
    def remove_diacritics(self, text: str) -> str:
        text = re.sub("[\u064B-\u065F]","",text)
        return text

    def normalize_alef(self, text: str) -> str:
        text = re.sub("[أإآٱ]","ا",text)
        return text

    def normalize_teh_marbuta(self, text: str) -> str:
        text = re.sub("ة","ه",text)
        return text
    
    def remove_any_non_arabic_or_number(self, text: str) -> str:
        text = re.sub("[^ا-ي0-9]+"," ",text).strip()
        return text
    
    def remove_repeated_characters(self, text: str) -> str:
        text = re.sub(r"(.)\1{2,}", r"\1\1", text) 
        return text

    def tokenize(self, text: str) -> List[str]:
        lst = text.split()
        return lst


    def preprocess(self, text: str):
        text1 = self.remove_diacritics(text)
        text2 = self.normalize_alef(text1)
        text3 = self.normalize_teh_marbuta(text2)
        text4 = self.remove_any_non_arabic_or_number(text3)
        text5 = self.remove_repeated_characters(text4)        
        
        lst = self.tokenize(text5)
        return(lst)