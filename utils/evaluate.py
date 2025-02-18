import numpy as np
import nltk
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# Uncomment the following line to download required NLTK data
# nltk.download('punkt')

SEP = "</SEP>"
DECIMAL = 6


def jaccard_similarity(text):
    """
    Calculate Jaccard similarity between two texts.

    Args:
        text (str): Two texts separated by SEP.

    Returns:
        float: Jaccard similarity score rounded to DECIMAL places.
    """
    text1, text2 = text.split(SEP)[0], text.split(SEP)[1]
    
    # Convert texts to lowercase and split into words
    text1 = np.array(text1.lower().split(" "))
    text2 = np.array(text2.lower().split(" "))

    # Calculate intersection and union
    intersection = len(np.intersect1d(text1, text2))
    union = len(np.union1d(text1, text2))

    # Calculate Jaccard similarity
    j_similarity = round(float(intersection / union), DECIMAL)

    return j_similarity


def cosine_similarity(text):
    """
    Calculate cosine similarity between two texts.

    Args:
        text (str): Two texts separated by SEP.

    Returns:
        float: Cosine similarity score rounded to DECIMAL places.
    """
    text1, text2 = text.split(SEP)[0], text.split(SEP)[1]
    
    # Convert texts to lowercase and split into words
    text1 = text1.lower().split(" ")
    text2 = text2.lower().split(" ")

    # Create a set of unique words from both texts
    unique_words = set(text1 + text2)

    # Create vectors for each text
    vector1 = np.array([text1.count(word) for word in unique_words])
    vector2 = np.array([text2.count(word) for word in unique_words])

    # Calculate dot product and magnitudes
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.sqrt(np.sum(vector1 ** 2))
    magnitude2 = np.sqrt(np.sum(vector2 ** 2))

    # Calculate cosine similarity
    cosine_sim = round(float(dot_product / (magnitude1 * magnitude2)), DECIMAL)

    return cosine_sim


def euclidean_distance(text):
    """
    Calculate Euclidean distance between two texts.

    Args:
        text (str): Two texts separated by SEP.

    Returns:
        float: Euclidean distance rounded to DECIMAL places.
    """
    text1, text2 = text.split(SEP)[0], text.split(SEP)[1]
    
    # Convert texts to lowercase and split into words
    text1 = text1.lower().split(" ")
    text2 = text2.lower().split(" ")

    # Create a set of unique words from both texts
    unique_words = set(text1 + text2)

    # Create vectors for each text
    vector1 = np.array([text1.count(word) for word in unique_words])
    vector2 = np.array([text2.count(word) for word in unique_words])

    # Calculate Euclidean distance
    euclidean_dist = round(float(np.linalg.norm(vector1 - vector2)), DECIMAL)

    return euclidean_dist


def bleu_score(text):
    """
    Calculate BLEU score between two texts.

    Args:
        text (str): Two texts separated by SEP.

    Returns:
        float: BLEU score rounded to DECIMAL places.
    """
    text1, text2 = text.split(SEP)[0], text.split(SEP)[1]
    
    # Tokenize the texts
    vector1 = word_tokenize(text1.lower())
    vector2 = word_tokenize(text2.lower())
    
    # Calculate BLEU score
    bleu = round(sentence_bleu([vector1], vector2), DECIMAL)
    
    return bleu
