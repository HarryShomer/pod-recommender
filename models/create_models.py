import gensim
from gensim.test.utils import datapath
from gensim.similarities.docsim import Similarity
import json
import process_data


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
    # TODO: How many topics???
    lda = gensim.models.LdaMulticore(corpus_tfidf, num_topics=57, id2word=dictionary, random_state=42)
    lda.save(datapath("lda_model.pkl"))


def lsi_model(corpus_tfidf, dictionary):
    """
    Create & save the LSI model

    :param corpus_tfidf: TF-IDF scores for each doc
    :param dictionary: Dictionary of all words

    :return: None
    """
    # TODO: How many topics???
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
    Create the models and calculate cosine similarity

    :return: 
    """
    data = process_data.create_model_data()
    print("Creating the models", end="", flush=True)

    transcript_tfidf, transcript_dict = get_tfidf(data['transcripts'])
    print(".", end="", flush=True)

    #lda_model(transcript_tfidf, transcript_dict)
    print(".", end="", flush=True)

    #lsi_model(transcript_tfidf, transcript_dict)
    print(". Done")


if __name__ == "__main__":
    run_analysis()
