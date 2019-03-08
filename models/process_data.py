"""
Module for processing the data for the model
"""
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import json

pd.options.mode.chained_assignment = None

# Saves time to just do it once
punctuation_table = str.maketrans({key: None for key in string.punctuation})
STOP_WORDS = set(stopwords.words('english'))


def clean_data(pod_df):
    """
    Fix the inputs and prepare for analysis.

    I searched for the word sponsor and these are basically the only relevant ones I
    could find. Sometimes the word sponsor is actually part of the pod description. 
    
    We attempt to strip:
    1. Sponsorships
    2. Dates
    3. Season & Episode #'s
    
    :param pod_df: DataFrame
    
    :return: cleaned DataFrame
    """
    sponsorships = [
        "Poynter's podcasts are sponsored by The City University of New York Graduate School of Journalism.",

        "sponsored by the International Anesthesia Research Society, which was founded by Dr. Ed Nemergut and Dr. Robert "
        "Thiele. It debuted in July 2009 with the broad goal of advancing graduate medical education in anesthesia. Since "
        "its inception as an experimental project, OpenAnesthesia™ has grown to be a comprehensive resource for anesthesiology "
        "residents and physicians worldwide.",

        "Co-sponsored by Synopsys and IEEE Security & Privacy.",

        "Cosponsored by the Publications Board and American Archivist Editorial Board of the Society of American Archivists (SAA)",

        "and the sponsorship of Swinburne Astronomy Online.",

        "Sponsored by DX Engineering.",

        "MASTERPIECE Studio is made possible by Viking Cruises and Farmers Insurance. Sponsors for MASTERPIECE on PBS are "
        "Viking Cruises, Farmers Insurance, and The MASTERPIECE Trust."
    ]

    # Below is an example for each pattern
    # 1. '(#\d+)': #35
    # 2. 'episode\s+\d+': episode 100
    # 3. 's\d+e\d+': s2e30
    # 4. 'season\s+\d+': season 32
    # 5. 'ep\s+\d+': ep 45
    # 6. 'ep\d+': ep45
    # 7. 'ep\.\d+': ep.45
    # 8. 'vol\.\s+\d+': vol. 648
    se_patterns = [r'(#\d+)', r'episode\s+\d+', r's\d+e\d+', r'season\s+\d+', r'ep\s+\d+', r'ep\d+', r'ep\.\d+',
                   r'vol\.\s+\d+']

    months = ["january", "february", "march", "april",
              "may", "june", "july", "august",
              "september", "october", "november", "december"]
    days = [r'sunday', r'monday', r'thursday', r'wednesday', r'thursday', r'friday', r'saturday']

    # Just drop 6 rows
    pod_df = pod_df.dropna()

    # standardize b4 we start
    pod_df['pod_description'] = pod_df['pod_description'].str.lower()
    pod_df['episode_name'] = pod_df['episode_name'].str.lower()
    pod_df['episode_description'] = pod_df['episode_description'].str.lower()
    pod_df['pod_name'] = pod_df['pod_name'].str.lower()

    for sponsor in sponsorships:
        pod_df['pod_description'] = pod_df['pod_description'].str.replace(sponsor, "")

    for pod_col in ['episode_name', 'episode_description']:
        for day in days:
            pod_df[pod_col] = pod_df[pod_col].str.replace(day, "")

        for pattern in se_patterns:
            pod_df[pod_col] = pod_df[pod_col].str.replace(pattern, "", regex=True)

        for month in months:
            # Examples of each:
            # January 23rd 2019
            # January 23rd, 2019
            # January 2019
            month_patterns = [f"{month}\s+\w+\s+\d+", f"{month}\s+\w+,\s+\d+", f"{month}\s+\d+"]
            for month_pattern in month_patterns:
                pod_df[pod_col] = pod_df[pod_col].str.replace(month_pattern, "", regex=True)

    return pod_df


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
    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())

    # Get rid of Stop words
    # Also get rid of punctuation
    filtered_words = [word.translate(punctuation_table) for word in tokens if word not in STOP_WORDS]

    # Lemmatize words
    wordnet_lemmatizer = WordNetLemmatizer()
    words_lemma = [wordnet_lemmatizer.lemmatize(word) for word in filtered_words]

    # Get rid of weird chars and encoding stuff
    words_lemma = [word.encode("ascii", "ignore").decode("utf-8") for word in words_lemma]
    words_lemma = [word for word in words_lemma if word not in ['-', '', ':', '|']]

    return words_lemma


def create_model_data():
    """
    Create the data for the analysis and deposit in a json file
    
    :return: None
    """
    df = pd.read_csv("../all_pod_episodes.csv", sep=',')
    df = clean_data(df)

    # ROWS: 324321
    pod_dict = df.to_dict("records")
    pod_lemmas = {'episode_name': [], 'pod_name': [], 'episode_description_tokens': [], 'episode_name_tokens': [],
                  'pod_name_tokens': [], 'pod_description_tokens': []}

    i = 1
    for row in pod_dict:
        pod_lemmas['episode_name'].append(row['episode_name'])
        pod_lemmas['pod_name'].append(row['pod_name'])
        for col in ['episode_name', 'episode_description', 'pod_description', 'pod_name']:
            pod_lemmas["_".join([col, "tokens"])].append(process_doc(row[col]))

        if i % 1000 == 0:
            print(i / 1000)
        i += 1

    with open("model_data.json", "w+") as file:
        json.dump(pod_lemmas, file)


def main():
    create_model_data()


if __name__ == "__main__":
    main()
