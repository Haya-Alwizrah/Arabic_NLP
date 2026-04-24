from preprocessing import ArabicPreprocessor

p = ArabicPreprocessor()

text = """
السَلَامُ عَليكم ورحمههه الله وبركاتههه
👑👑👑
✨🔟

#الالاب @hhggf
https:haygghf
"""
d = p.preprocess(text)
print(d)