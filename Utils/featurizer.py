import nltk
import re
from nltk.tokenize import word_tokenize
import spacy

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

def combiner_function(text):
    # Here the `all_feats` dict should contain the features -- the key should be the feature name,
    # and the value is the feature value.  See `simple_featurize` for an example.
    # at the moment, all 4 of: bag of words and your 3 original features are handed off to the combined model
    # update the values within [bag_of_words, feature1, feature2, feature3] to change this.
    all_feats={}
    features_used = [get_length, binary_bow_featurize, question_word_diction, pos_tag,
                    flesch_kincaid_grade_level, calculate_syntactic_complexity]
    #print("Used features: ")
    #print(", ".join(map(str, features_used)))
    for feature in features_used:
        all_feats.update(feature(text))
    return all_feats