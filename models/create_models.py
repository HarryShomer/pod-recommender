import os
import json
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.similarities import MatrixSimilarity
from sklearn.model_selection import train_test_split
import process_data

plt.style.use('ggplot')

MAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))

PODCASTS = {
    "codeswitch": "Society & Culture",
    "embedded": "News & Politics",
    "npr-politics-podcast": "News & Politics",
    "hidden-brain": "Science & Medicine",
    "invisibilia": "Science & Medicine",
    "wow-in-the-world": "Science & Medicine",
    "planet-money": "Business",
    "the-indicator-from-planet-money": "Business",
    "Learning English": "Language",
    "Limetown": "Fiction",
    "Mable": "Fiction",
    "Mogul The Life and Death of Chris Lighty": "Investigative Journalism",
    "S-Town": "Investigative Journalism",
    "Serial": "Investigative Journalism"
}

MAX_TOPICS = 10


def get_tfidf(docs, dictionary):
    """
    Get the TF-IDF transformed data

    :param docs: given descriptor

    :return: tfidf model
    """
    # Make to bag of words
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

    # TF-IDF
    tfidf = gensim.models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    return corpus_tfidf


def create_lda_model(corpus_tfidf, dictionary, topics):
    """
    Create & save the LDA model
    
    :param corpus_tfidf: TF-IDF scores for each doc
    :param dictionary: Dictionary of all words
    :param topics: # of topics
    
    :return: None
    """
    if not os.path.isdir(os.path.join(MAIN_PATH, "lda")):
        os.mkdir(os.path.join(MAIN_PATH, "lda"))

    lda = gensim.models.LdaMulticore(corpus_tfidf, num_topics=topics, id2word=dictionary, random_state=42)
    lda.save(os.path.join(MAIN_PATH, "lda", f"lda_{topics}.model"))


def create_lsi_model(corpus_tfidf, dictionary, topics):
    """
    Create & save the LSI model

    :param corpus_tfidf: TF-IDF scores for each doc
    :param dictionary: Dictionary of all words
    :param topics: # of topics

    :return: None
    """
    if not os.path.isdir(os.path.join(MAIN_PATH, "lsi")):
        os.mkdir(os.path.join(MAIN_PATH, "lsi"))

    lsi = gensim.models.LsiModel(corpus_tfidf, num_topics=topics, id2word=dictionary)
    lsi.save(os.path.join(MAIN_PATH, "lsi", f"lsi_{topics}.model"))


def calc_similarity(model_data, corpus, dictionary):
    """
    Calculate the similar podcasts for all podcasts and both methods
    
    :param corpus: 
    :param dictionary: 
    
    :return: 
    """
    preds = {}
    for metric in ['lda', 'lsi']:
        for i in range(1, 6):
            preds[f"{metric}_{i}_similarity_correct"] = {}
            preds[f"{metric}_{i}_similarity_same"] = {}

    print("Calculating the metrics", end="", flush=True)

    for topics in range(5, MAX_TOPICS+1):
        results = []
        lda = gensim.models.LdaModel.load(os.path.join(MAIN_PATH, "lda", f"lda_{topics}.model"))
        lsi = gensim.models.LsiModel.load(os.path.join(MAIN_PATH, "lsi", f"lsi_{topics}.model"))

        lda_index = MatrixSimilarity(lda[corpus], num_best=5, num_features=len(dictionary))
        lsi_index = MatrixSimilarity(lsi[corpus], num_best=5, num_features=len(dictionary))

        for pod, lda_similarities, lsi_similarities in zip(model_data, lda_index[corpus], lsi_index[corpus]):
            episode_sim = {}
            for sim_metric in [["lda", lda_similarities], ["lsi", lsi_similarities]]:
                for sim in range(len(sim_metric[1])):
                    base_col = f'{sim_metric[0]}_{sim+1}_similarity'
                    sim_pod = model_data[sim_metric[1][sim][0]]['podcast']

                    # Is it "correct"
                    episode_sim[f'{base_col}_correct'] = PODCASTS[pod['podcast']] == PODCASTS[sim_pod]
                    # If same pod
                    episode_sim[f'{base_col}_same'] = pod['podcast'] == sim_pod

            results.append(episode_sim)

        # Get the percentages for each
        df = pd.DataFrame(results).sum() / pd.DataFrame(results).shape[0]
        for col in preds:
            preds[col][topics] = df[col]

        print(".", end="", flush=True)

    with open("results.json", "w") as file:
        json.dump(preds, file, indent=4)

    print(". Done")


def run_analysis(data, model_data):
    """
    Create the models and calculate cosine similarity

    :return: None
    """
    print("Creating the models", end="", flush=True)

    # Map all words in both training and testing set
    dictionary = gensim.corpora.Dictionary([podcast['transcript'] for podcast in data])

    """
    # Get tfidf for training data
    train_tfidf = get_tfidf([podcast['transcript'] for podcast in model_data['train']], dictionary)
    print(".", end="", flush=True)

    for topics in range(5, MAX_TOPICS+1):
        create_lda_model(train_tfidf, dictionary, topics)
        create_lsi_model(train_tfidf, dictionary, topics)
        print(".", end="", flush=True)
    print(". Done")
    """

    # Get results for test data
    test_tfidf = get_tfidf([podcast['transcript'] for podcast in model_data['test']], dictionary)
    calc_similarity(model_data['test'], test_tfidf, dictionary)


def create_viz():
    """
    """
    if not os.path.isdir(os.path.join(MAIN_PATH, "viz")):
        os.mkdir(os.path.join(MAIN_PATH, "viz"))

    with open(os.path.join(MAIN_PATH, "results.json"), "r") as file:
        results = json.load(file)

    ###################################
    # 1. Create a separate viz for each
    ###################################
    for model_type in ["lda", "lsi"]:
        for results_type in ['correct', 'same']:
            plt.figure()

            for col in [f"{model_type}_{i}_similarity_{results_type}" for i in range(1, 6)]:
                plt.plot(np.arange(5, MAX_TOPICS+1), [results[col][i] for i in results[col]], label=col)

            plt.title(f"{model_type} {results_type} Accuracy")
            plt.xlabel("# of Topics")
            plt.ylabel("Accuracy")
            plt.legend(loc="lower right", prop={'size': 8})
            plt.savefig(os.path.join(MAIN_PATH, "viz", f"{model_type}_{results_type}_plot.png"))

    acc_avgs = {}

    #######################################
    # 2. Avg out top 5 and place on one viz
    #######################################
    plt.figure()
    for model_type in ["lda", "lsi"]:
        for results_type in ['correct', 'same']:
            acc_avgs[f"{model_type}_{results_type}"] = []

            # Get the cumulative top 5 average for each
            for topics in range(5, MAX_TOPICS+1):
                model_accs = []
                for result_num in range(1, 6):
                    model_accs.append(results[f"{model_type}_{result_num}_similarity_{results_type}"][str(topics)])

                acc_avgs[f"{model_type}_{results_type}"].append(np.mean(model_accs))

            plt.plot(np.arange(5, MAX_TOPICS+1), acc_avgs[f"{model_type}_{results_type}"], label=f"{model_type}_{results_type}")

    plt.title("Top 5 Accuracy")
    plt.xlabel("# of Topics")
    plt.ylabel("Top 5 Acc%")
    plt.legend(loc="lower right", prop={'size': 8})
    plt.savefig(os.path.join(MAIN_PATH, "viz", "Top_5_plot.png"))

    #######################################
    # 3. Correct% - Same% = Uniqueness
    #######################################
    plt.figure()
    plt.plot(np.arange(5, MAX_TOPICS+1), np.array(acc_avgs["lda_correct"]) - np.array(acc_avgs["lda_same"]), label="lda")
    plt.plot(np.arange(5, MAX_TOPICS+1), np.array(acc_avgs["lsi_correct"]) - np.array(acc_avgs["lsi_same"]), label="lsi")
    plt.title("Top 5 Uniqueness")
    plt.xlabel("# of Topics")
    plt.ylabel("Correct% - Same%")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(MAIN_PATH, "viz", "Uniqueness_plot.png"))


def main():
    data = process_data.create_model_data()

    model_data = {}
    model_data['train'], model_data['test'], _, _ = train_test_split(data, [i for i in range(len(data))], test_size=.2,
                                                                     random_state=42)
    run_analysis(data, model_data)
    create_viz()


if __name__ == "__main__":
    main()
