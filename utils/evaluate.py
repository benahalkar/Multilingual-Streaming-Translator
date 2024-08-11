
import numpy as np
import nltk
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# nltk.download('punkt')

SEP = "</SEP>"
DECIMAL = 6


def jaccard_similarity(text):
    text1, text2 = text.split(SEP)[0], text.split(SEP)[1]
    
    text1 = np.array(text1.lower().split(" "))
    text2 = np.array(text2.lower().split(" "))

    intersection = len(np.intersect1d(text1, text2))
    union = len(np.union1d(text1, text2))

    j_similarity = round(float(intersection/union), DECIMAL)

    return j_similarity



def cosine_similarity(text):
    text1, text2 = text.split(SEP)[0], text.split(SEP)[1]
    
    text1 = text1.lower().split(" ")
    text2 = text2.lower().split(" ")

    unique_words = set(text1 + text2)

    vector1 = np.array([text1.count(word) for word in unique_words])
    vector2 = np.array([text2.count(word) for word in unique_words])

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.sqrt(np.sum(vector1 ** 2))
    magnitude2 = np.sqrt(np.sum(vector2 ** 2))

    cosine_sim = round(float(dot_product / (magnitude1 * magnitude2)), DECIMAL)

    return cosine_sim



def eucledian_distance(text):
    text1, text2 = text.split(SEP)[0], text.split(SEP)[1]
    
    text1 = text1.lower().split(" ")
    text2 = text2.lower().split(" ")

    unique_words = set(text1 + text2)

    vector1 = np.array([text1.count(word) for word in unique_words])
    vector2 = np.array([text2.count(word) for word in unique_words])

    eucledian_dist = round(float(np.linalg.norm(vector1 - vector2)), DECIMAL)

    return eucledian_dist



def bleu_score(text):
    text1, text2 = text.split(SEP)[0], text.split(SEP)[1]
    
    vector1 = word_tokenize(text1.lower())
    vector2 = word_tokenize(text2.lower())
    
    bleu_score = round(sentence_bleu([vector1], vector2), DECIMAL)
    
    return bleu_score