import gensim
from gensim.test.utils import datapath
from gensim.similarities.docsim import Similarity
import os
import json


def get_tfidf(docs):
    """
    Get the TF-IDF transformed data

    :param docs: given descriptor

    :return: tfidf model, dictionary
    """
    # Map all words
    dictionary = gensim.corpora.Dictionary(docs)

    # Make to bag of words
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

    # TF-IDF
    tfidf = gensim.models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    return corpus_tfidf, dictionary


def lda_model(corpus_tfidf, dictionary):
    """
    Create & save the LDA model
    
    :param corpus_tfidf: TF-IDF scores for each doc
    :param dictionary: Dictionary of all words
    
    :return: None
    """
    lda = gensim.models.LdaMulticore(corpus_tfidf, num_topics=57, id2word=dictionary, random_state=42)
    lda.save(datapath("lda_model.pkl"))


def lsi_model(corpus_tfidf, dictionary):
    """
    Create & save the LSI model

    :param corpus_tfidf: TF-IDF scores for each doc
    :param dictionary: Dictionary of all words

    :return: None
    """
    lsi = gensim.models.LsiModel(corpus_tfidf, num_topics=57, id2word=dictionary)
    lsi.save(datapath("lsi_model.pkl"))

    # vector = model[common_corpus[4]]


def log_entropy(corpus_tfidf, model_data):
    pass


def cosine_similarity(corpus, dictionary):
    # index = Similarity(index_temp, corpus, num_features=len(dictionary))
    pass


def run_analysis():
    """
    16 General categories
    57 Nested Categories

    :return: 
    """
    model_cols = ['episode_description_tokens', 'episode_name_tokens', 'pod_name_tokens', 'pod_description_tokens']

    with open("model_data.json") as file:
        model_data = json.load(file)

    from time import time

    # Processed data
    transformed_data = {key: {} for key in model_cols}
    transformed_data['pod_name'] = model_data['pod_name']
    transformed_data['episode_name'] = model_data['episode_name']

    for col in model_cols:
        #print("Getting tf-idf for", col)
        transformed_data[col]['tfidf'], transformed_data[col]['dictionary'] = get_tfidf(model_data[col])

        start = time()
        print("Fitting LDA for", col)
        lda_model(transformed_data[col]['tfidf'], transformed_data[col]['dictionary'])
        print("Time spent", time() - start)

        start = time()
        print("Fitting LSI for", col)
        lsi_model(transformed_data[col]['tfidf'], transformed_data[col]['dictionary'])
        print("Time spent", time() - start)


if __name__ == "__main__":
    run_analysis()
