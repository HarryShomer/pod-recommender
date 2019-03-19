"""
Module for processing the data for the model
"""
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import re

# Saves time to just do it once
punctuation_table = str.maketrans({key: None for key in string.punctuation})
STOP_WORDS = set(stopwords.words('english'))


# TODO: What else should we look for?
def clean_data(text, pod):
    """
    Clean the data for analysis. Get rid of unnecessary stuff
    
    :param text: String of text
    :param pod: Specific podcast
    
    :return: cleaned text
    """
    text = text.lower()

    # Various names of hosts for podcasts obtained from websites
    hosts = {
        "planet-money": "Ailsa Chang Cardiff Garcia Jacob Goldstein Noel King Kenny Malone Robert Smith Stacey Vanek Smith",
        "the-indicator-from-planet-money": "Ailsa Chang Cardiff Garcia Jacob Goldstein Noel King Kenny Malone Robert Smith Stacey Vanek Smith",
        "hidden-brain": "Shankar Vedantam",
        "npr-politics-podcast": "",
        "invisibilia": "Alix Spiegel Hanna Rosin Lulu Miller",
        "embedded": "Kelly McEvers",
        "codeswitch": "Shereen Marisol Meraji Gene Demby Adrian Florido Karen Grigsby Bates Leah Donnella "
                      "Maria Paz Gutierrez Tiara Jenkins Kat Chow",
        "wow-in-the-world": "Guy Raz Mindy Thomas",
    }

    # 1. \(.+?(?=\))\) = Match everything in parentheses
    # 2. \[.+?(?=\])\] = Match everything in brackets
    # 3. HOST = The words 'host' shows up here and there
    # 4. \w+: = Always a name when like this
    patterns = [r"\(.+?(?=\))\)", r"\[.+?(?=\])\]", r"host", r"\s+\w+:"]
    for pattern in patterns:
        text = re.sub(pattern, " ", text)

    # Remove host names from appropriate podcasts
    for host in hosts.get(pod, "").split():
        text = text.replace(host.lower(), " ")

    return text


def process_doc(text, pod):
    """
    Process a given string of text:
    1. "Clean" text -> Get rid of unnecessary words 
    2. Split into individual words
    3. Convert to lowercase
    4. Get rid of stop words
    5. No punctuation
    6. Lemmatize
    
    :param text: String of text
    :param pod: Specific podcast
    
    :return: List of lemmatized words
    """
    text = clean_data(text, pod)
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
    Create the data for the analysis 
    
    :return: Dict of data
    """
    main_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "podcast_data")
    pods = {"transcripts": [], "pods": [], "episodes": []}

    print("Processing the episode transcripts", end="", flush=True)

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
                        pods['transcripts'].append(process_doc(file.read(), item))
                        pods['pods'].append(item)
                        pods["episodes"].append(episode[:episode.rfind(".")])
    print(" Done")

    return pods


def main():
    create_model_data()
    #pass

if __name__ == "__main__":
    main()
