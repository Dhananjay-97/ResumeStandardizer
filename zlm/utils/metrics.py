'''
-----------------------------------------------------------------------
File: metrics.py
Creation Time: Jan 13th 2024, 8:42 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
'''
import re
import json
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise
from zlm.utils.utils import key_value_chunking

import nltk
import os

# Get script directory, so we can resolve a local filepath
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
nltk_data_path = os.path.normpath(os.path.abspath(os.path.join(script_dir, 'nltk_data'))).strip() #added normalize abs path

#Manually insert at the 0 position, so it loads that first
if nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_path) #Add at the start!

# Verify that the filepath is actually downloaded and able to load that filepath.

try:
    print(f"NLTK data path: {nltk.data.path}")  # Verify that local path is in the list
    #Added try catch blocks and more detailed print statements for debugging

    print("Contents of nltk_data directory:")
    for item in os.listdir(nltk_data_path):  #Using nltk_data_path directly
        print(f"- {item}") #print items

        if os.path.isdir(os.path.join(nltk_data_path,item)): #if they are dir, print their contents too!
             print(f"  Contents of {item}:")
             for sub_item in os.listdir(os.path.join(nltk_data_path,item)):
                  print(f"    - {sub_item}")

except FileNotFoundError as e:
    print(f"Error: nltk_data directory not found: {e}") #Detailed exception info
except Exception as e:
    print(f"Error listing directory contents: {e}") #Detailed exception info

 #This is where all the magic is happening

try:
    #Now after setting up everything, download it IF it is not downloaded. It is already commented for production after one time use
    #Ensure these downloads are before any of the methods that use nltk
    if not os.path.exists(os.path.join(nltk_data_path,'tokenizers','punkt','english.pickle')): #check if file exists

        nltk.download('punkt', download_dir=nltk_data_path)
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
        nltk.download('stopwords', download_dir=nltk_data_path)

except Exception as e:
    print(f"There was an issue downloading the nltk library here is the error {e}")

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

def remove_urls(list_of_strings):
    """Removes strings containing URLs from a list using regular expressions."""
    filtered_list = [string for string in list_of_strings if not re.search(r"https?://\S+", string)]
    return filtered_list

def overlap_coefficient(document1: str, document2: str) -> float:
    """Calculate the overlap coefficient between two documents.

    The overlap coefficient is a measure of the similarity between two sets,
    and is defined as the size of the intersection divided by the smaller
    of the size of the two sets.

    Args:
        document1 (str): The first document.
        document2 (str): The second document.

    Returns:
        float: The overlap coefficient between the two documents.
    """
    # List the unique words in a document
    words_in_document1 = set(normalize_text(document1))
    words_in_document2 = set(normalize_text(document2))

    # Find the intersection of words list of document1 & document2
    intersection = words_in_document1.intersection(words_in_document2)

    # Calculate overlap coefficient
    try:
        overlap_coefficient = float(len(intersection)) / min(len(words_in_document1), len(words_in_document2))
    except ZeroDivisionError:
        overlap_coefficient = 0.0

    return overlap_coefficient

def jaccard_similarity(document1: str, document2: str) -> float:
    """Calculate the Jaccard similarity between two documents.

    The Jaccard similarity is a measure of the similarity between two sets,
    and is defined as the size of the intersection divided by the size of
    the union of the two sets.

    Args:
        document1 (str): The first document.
        document2 (str): The second document.

    Returns:
        float: The Jaccard similarity between the two documents.
    """
    # List the unique words in a document
    words_in_document1 = set(normalize_text(document1))
    words_in_document2 = set(normalize_text(document2))

    # Find the intersection of words list of document1 & document2
    intersection = words_in_document1.intersection(words_in_document2)

    # Find the union of words list of document1 & document2
    union = words_in_document1.union(words_in_document2)

    # Calculate Jaccard similarity score
    try:
        jaccard_similarity = float(len(intersection)) / len(union)
    except ZeroDivisionError:
        jaccard_similarity = 0.0

    return jaccard_similarity

def cosine_similarity(document1: str, document2: str) -> float:
    """Calculate the cosine similarity between two documents.

    Args:
        document1 (str): The first document.
        document2 (str): The second document.

    Returns:
        float: The cosine similarity between the two documents.
    """
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the documents into TF-IDF vectors
    vectors = vectorizer.fit_transform([document1, document2])

    cosine_similarity_score = pairwise.cosine_similarity(vectors[0], vectors[1])
    # Calculate the cosine similarity between the two vectors
    # cosine_similarity = np.dot(vectors[0], vectors[1].T) / (np.linalg.norm(vectors[0].toarray()) * np.linalg.norm(vectors[1].toarray()))

    return cosine_similarity_score.item()

def vector_embedding_similarity(llm, document1: str, document2: str) -> float:
    document1 = key_value_chunking(json.loads(document1))
    document2 = key_value_chunking(json.loads(document2))

    emb_1 = llm.get_embedding(document1, task_type="retrieval_query")
    emb_2 = llm.get_embedding(document2, task_type="retrieval_query")

    df1 = pd.DataFrame(emb_1.embedding.to_list())
    df2 = pd.DataFrame(emb_2.embedding.to_list())

    emb_sem = pairwise.cosine_similarity(df1, df2)

    return emb_sem.mean()


    pass

def normalize_text(text: str) -> list:
    """Normalize the input text.

    This function tokenizes the text, removes stopwords and punctuations,
    and applies stemming.

    Args:
        text (str): The text to normalize.

    Returns:
        list: The list of normalized words.
    """
    try:
        stop_words = set(stopwords.words('english'))
        words = [word.lower() for word in re.findall(r'\b\w+\b', text) if word.lower() not in stop_words]
        return words #Return the words

    except Exception as e:
        print(f"Error in normalize_text: {e}")
        return []  # Return an empty list in case of an error