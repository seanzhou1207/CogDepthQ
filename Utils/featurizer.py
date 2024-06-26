import nltk
import re
from nltk.tokenize import word_tokenize
import spacy
import random

nltk.download('punkt')
#!python -m spacy download en_core_web_sm   # Need to download this separately

# Features
def binary_bow_featurize(text):
    feats = {}
    words = nltk.word_tokenize(text)

    for word in words:
        word=word.lower()
        feats[word]=1
            
    return feats

def bow_featurize(text):
    feats = {}
    words = nltk.word_tokenize(text)
    for word in words:
        word=word.lower()
        if word in feats:
            feats[word] += 1
        else:
            feats[word] = 1
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

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

def get_max_depth(token, current_depth=0):
    """Recursively find the maximum depth of the dependency tree."""
    if not list(token.children):
        return current_depth
    return max(get_max_depth(child, current_depth + 1) for child in token.children)

def calculate_syntactic_complexity(sentence):
    """Calculate the syntactic complexity of a sentence by finding the max depth of its parse tree."""
    feats = {}
    doc = nlp(sentence)
    root = next(tok for tok in doc if tok.dep_ == 'ROOT')  # Find the root of the sentence
    max_depth = get_max_depth(root)

    feats["syn_complexity"] = max_depth
    return feats

def question_word_diction(text):
        question_word_list = ['what', 'where', 'when','how','why','did','do','does',
                'have','has','am','is','are','can','could','may','would','will','should',
"didn't","doesn't","haven't","isn't","aren't","can't","couldn't","wouldn't","won't","shouldn't",'?']
        feats = {}
        words = nltk.word_tokenize(text)
        for word in words:
            if word.lower() in feats:
                  feats[word.lower()] += 1
            elif word.lower() in question_word_list:
                  feats[word.lower()] = 1
        return feats

def pos_tag(text):
    feats = {"VB":0,"VBD":0,"VBG":0,"VBN":0,"VBP":0,"VBZ":0,"JJ":0,"JJR":0,"JJS":0}

    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    pos_tags_seq = [tag for word, tag in tags]
    for pos_tag in pos_tags_seq:
        if pos_tag in feats:
            feats[pos_tag] += 1
    return feats

def bow_featurize_vb_adj(text):
    feats = {}

    words = nltk.word_tokenize(text)
    words_tags = nltk.pos_tag(words)

    for word_tag in words_tags:
        word = word_tag[0]
        tag = word_tag[1]
        word=word.lower()
        if tag in ["VB","VBD","VBG","VBN","VBP","VBZ","JJ","JJR","JJS"]:
            if word in feats:
                feats[word] += 1
            else:
                feats[word] = 1
    return feats


# Owen
# 1) Afinn sentiment (give a sentiment score from -7 to 7): No effect
from afinn import Afinn
def afinn_sentiment(text):
    # Here the `feats` dict should contain the features -- the key should be the feature name,
    # and the value is the feature value.  See `simple_featurize` for an example.
    feats = {}
    afinn = Afinn()
    sentences = nltk.sent_tokenize(text)
    for i, sentence in enumerate(sentences):
      feats[i] = afinn.score(sentence)
    return feats

from collections import Counter
# 2） Bigram: No effect
def bigram(text):
    # Here the `feats` dict should contain the features -- the key should be the feature name,
    # and the value is the feature value.  See `simple_featurize` for an example.
    feats = {}
    words = nltk.word_tokenize(text)
    trigrams = [' '.join(tg) for tg in list(nltk.bigrams(words))]
    feats = dict(Counter(trigrams))
    return feats

# 3) Vader sentiment (give 4 features 'pos', 'neg', 'neu', and 'compound'): No effect
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
def Vader_sentiment(text):
    feats = {}
    feats["pos"] = 0
    feats["neg"] = 0
    feats["neu"] = 0
    feats["compound"] = 0
    sentences = nltk.sent_tokenize(text)
    sia = SentimentIntensityAnalyzer()
    for sentence in sentences:
        score = sia.polarity_scores(sentence)
        feats["pos"] += score["pos"]
        feats["neg"] += score ["neg"]
        feats["neu"] += score ["neu"]
        feats["compound"] += score ["compound"]
    return feats

# 4) LIWC score (give various features such as counts of "pronoun", "i", "ppron", etc) :lower accuracy
import liwc
def liwc_pos_type(text):
    # Here the `feats` dict should contain the features -- the key should be the feature name,
    # and the value is the feature value.  See `simple_featurize` for an example.
    feats = {}
    words = nltk.word_tokenize(text)
    dictionary, category_names = liwc.read_dic("LIWC2007_English100131.dic")

    for word in words:
        word=word.lower()
        if word in dictionary:
            for category in dictionary[word]:
                if category in feats:
                    feats[category] += 1
                else:
                    feats[category] = 1

    return feats


def combiner_function(text):
    # Here the `all_feats` dict should contain the features -- the key should be the feature name,
    # and the value is the feature value.  See `simple_featurize` for an example.
    # at the moment, all 4 of: bag of words and your 3 original features are handed off to the combined model
    # update the values within [bag_of_words, feature1, feature2, feature3] to change this.
    all_feats={}
    features_used = [bow_featurize, get_length, question_word_diction, #bow_featurize_vb_adj, pos_tag, #binary_bow_featurize,pos_tag, 
                    calculate_syntactic_complexity, flesch_kincaid_grade_level] #flesch_kincaid_grade_level, bigram]
    #print("Used features: ")
    #print(", ".join(map(str, features_used)))
    for feature in features_used:
        #dice = random.randint(0, 1)
        #if dice == 0:
        all_feats.update(feature(text))
        #print("used: ")
        #print(str(feature))
        #else:
        #    continue
    return all_feats