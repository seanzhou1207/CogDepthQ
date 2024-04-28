import nltk
import re
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Features
def binary_bow_featurize(text):
    feats = {}
    words = nltk.word_tokenize(text)

    for word in words:
        word=word.lower()
        feats[word]=1
            
    return feats

def get_length(text):
    """
    Get length of questions
    """
    feats = {}
    tokens = nltk.word_tokenize(text)

    full_len = len(tokens)
    feats["sent_length"] = full_len
    return feats

# GPT Code
def count_syllables(word):
    # Regex pattern to count the number of vowel sequences in a word, which approximates syllable count.
    return len(re.findall(r'[aeiouyAEIOUY]+', word))

# GPT Code
def flesch_kincaid_grade_level(text):
    feats = {}

    sentences = re.split(r'[.!?]', text)
    sentence_count = sum(bool(x.strip()) for x in sentences)  # Count non-empty sentences
    words = re.findall(r'\b\w+\b', text)
    syllable_count = sum(count_syllables(word) for word in words)
    word_count = len(words)
    
    # Flesch-Kincaid Grade Level formula
    fk_grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
    feats["fk_grade"] = fk_grade
    return feats

def combiner_function(text):
    # Here the `all_feats` dict should contain the features -- the key should be the feature name,
    # and the value is the feature value.  See `simple_featurize` for an example.
    # at the moment, all 4 of: bag of words and your 3 original features are handed off to the combined model
    # update the values within [bag_of_words, feature1, feature2, feature3] to change this.
    all_feats={}
    for feature in [binary_bow_featurize, get_length,flesch_kincaid_grade_level]:
        all_feats.update(feature(text))
    return all_feats