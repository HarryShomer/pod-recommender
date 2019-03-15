"""
Module for processing the data for the model
"""
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import json
import os

pd.options.mode.chained_assignment = None

# Saves time to just do it once
punctuation_table = str.maketrans({key: None for key in string.punctuation})
STOP_WORDS = set(stopwords.words('english'))


# TODO: What should we look for?
def clean_data():
    """
    :return: 
    """
    pass


def process_doc(text):
    """
    Process a given string of text:
    1. Split into individual words
    2. Convert to lowercase
    3. Get rid of stop words
    4. No punctuation
    5. Lemmatize
    
    :param text: String of text
    
    :return: List of lemmatized words
    """
    tokens = word_tokenize(text)

    # Get rid of Stop words and punctuation
    # Also make lowercase
    filtered_words = [word.translate(punctuation_table).lower() for word in tokens if word not in STOP_WORDS]

    wordnet_lemmatizer = WordNetLemmatizer()
    words_lemma = [wordnet_lemmatizer.lemmatize(word) for word in filtered_words]

    # Get rid of weird chars and encoding stuff
    #words_lemma = [word.encode("ascii", "ignore").decode("utf-8") for word in words_lemma]

    return words_lemma


def create_model_data():
    """
    Create the data for the analysis and deposit in a json file
    
    :return: None
    """
    main_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "podcast_data")
    pods = {}

    print("Processing the transcripts", end="", flush=True)

    # Get all the podcasts
    for pod_type in ['assorted', 'npr']:
        for item in os.listdir(os.path.join(main_dir, pod_type)):
            if os.path.isdir(os.path.join(main_dir, pod_type, item)):
                print(".", end="", flush=True)
                pods[item] = {}

                # Now get all epsiodes for the podcast
                for episode in os.listdir(os.path.join(main_dir, pod_type, item)):
                    with open(os.path.join(main_dir, pod_type, item, episode), 'r', 
                              encoding='utf-8', errors='ignore') as file:
                        pods[item][episode[:episode.rfind(".")]] = process_doc(file.read())

    print(" Done")


def main():
    create_model_data()

if __name__ == "__main__":
    main()
