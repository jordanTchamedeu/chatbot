import nltk
import numpy as np
nltk.download('punkt')
from  nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()

def tokenize(text):
    return nltk.word_tokenize(text)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_words, all_words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_words]
    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

sentence=["hello","you","i am fine"]

words=["hi","hello","I","you","Bye","thank you","cool"]
bag=bag_of_words(sentence,words)
print(bag)