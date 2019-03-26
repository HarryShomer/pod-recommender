"""
This module creates a number of different LDA and LSI models. Each model differs in the # of numbers. The range of numbers
is between the MIN_TOPICS and MAX_TOPICS global variables. 

For each episode we obtained the top 5 recommendations for episodes in the entire dataset. In order to determine how well 
those recommendations are the following strategy was used: For each podcast you can look up the associated `genre` that 
it belongs to (it may say on the website or you can check what google or itunes brings up). We can then compare the
genre for the podcast and the recommendations. The assumption here is that the model should recommend podcasts in the
same genre. This was done for each podcast and can be found in the PODCAST_CATS global variable. 

That is how we check for `accuracy`. Another consideration was if the recommendation was an episode from the same 
podcast. Another assumption made is that while it makes sense to recommend episodes from the same podcast, it is an
easy and boring answer. Therefore we should place more weight for the models that recommend different podcasts. We 
therefore calculate this as well. From this I calculated a `uniqueness` measure. This is just accuracy - sameness. 

All the graphs with the results can be found in the /models/viz folder. 
"""
import os
import json
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.similarities import MatrixSimilarity
import process_data

plt.style.use('ggplot')

MAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))

PODCASTS_CATS = {
    "codeswitch": "Society & Culture",
    "this-american-life": "Society & Culture",
    "embedded": "News & Politics",
    "npr-politics-podcast": "News & Politics",
    "hidden-brain": "Science & Medicine",
    "invisibilia": "Science & Medicine",
    "wow-in-the-world": "Science & Medicine",
    "planet-money": "Business",
    "the-indicator-from-planet-money": "Business",
    "Limetown": "Fiction",
    "Mable": "Fiction",
    "Mogul The Life and Death of Chris Lighty": "Investigative Journalism",
    "S-Town": "Investigative Journalism",
    "Serial": "Investigative Journalism"
}

# MAX and MIN number of topics we should use to build the models
MIN_TOPICS = 1
MAX_TOPICS = 14


def get_tfidf(docs, dictionary):
    """
    Get the TF-IDF transformed data

    :param docs: given descriptor
    :param dictionary: Mapping of words to #

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
    Get the top 5 similar podcasts episodes for each episode in the test set. We then use this to calculate 2 metrics,
    the % `correct` and `same`. `correct` means the two belong in the same category (see PODCAST above). `same` means 
    that the recommend episode belongs to the same podcast. 
    
    All the results are then stored in the `results.json` file. 
    
    :param model_data: Data used for creating the models
    :param corpus: Test tfidf
    :param dictionary: Mapping of words to #s 
    
    :return: 
    """
    print("Calculating the metrics", end="", flush=True)

    # Create structure to hold all the data
    preds = {}
    for metric in ['lda', 'lsi']:
        for i in range(1, 6):
            preds[f"{metric}_{i}_similarity_correct"] = {}
            preds[f"{metric}_{i}_similarity_same"] = {}

    # Test each of the 2 types of model from the # of topics used
    for topics in range(MIN_TOPICS, MAX_TOPICS+1):
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
                    episode_sim[f'{base_col}_correct'] = PODCASTS_CATS[pod['podcast']] == PODCASTS_CATS[sim_pod]
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


def run_analysis(data):
    """
    Create the models and calculate cosine similarity
    
    :param data: model data

    :return: None
    """
    print("Creating the models", end="", flush=True)

    # Map all words in both training and testing set
    dictionary = gensim.corpora.Dictionary([podcast['transcript'] for podcast in data])

    # Get tfidf for training data
    tfidf = get_tfidf([podcast['transcript'] for podcast in data], dictionary)

    # Create all the models
    for topics in range(MIN_TOPICS, MAX_TOPICS+1):
        create_lda_model(tfidf, dictionary, topics)
        create_lsi_model(tfidf, dictionary, topics)
        print(".", end="", flush=True)
    print(" Done")

    # Get results for test the data
    calc_similarity(data, tfidf, dictionary)


def create_viz():
    """
    This functions creates 3 types visualizations: 
    
    1. For each possible model (lda/lsi from MIN_TOPICS to MAX_TOPICS) the `accuracy` and `sameness` for each of the 
       5 recommendations individually. ['lda_correct.png', 'lda_same.png', 'lsi_correct.png', 'lsi_same.png']
    2. For each possible model (lda/lsi from MIN_TOPICS to MAX_TOPICS) the `accuracy` and `sameness` for the top 5
       recommendations as a whole. ['top_5.png'].
    3. I attempted to model how unique the models were by subtracting the accuracy from the sameness. This will give
       us how often they recommended podcasts that were not belonging to the same podcast. 
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
                plt.plot(np.arange(MIN_TOPICS, MAX_TOPICS+1), [results[col][i] for i in results[col]], label=col)

            plt.title(f"{model_type} {results_type} Accuracy")
            plt.xlabel("# of Topics")
            plt.ylabel("Accuracy")
            plt.legend(loc="lower left", prop={'size': 8})
            plt.savefig(os.path.join(MAIN_PATH, "viz", f"{model_type}_{results_type}.png"))

    acc_avgs = {}

    #######################################
    # 2. Avg out top 5 and place on one viz
    #######################################
    plt.figure()
    for model_type in ["lda", "lsi"]:
        for results_type in ['correct', 'same']:
            acc_avgs[f"{model_type}_{results_type}"] = []

            # Get the cumulative top 5 average for each
            for topics in range(MIN_TOPICS, MAX_TOPICS+1):
                model_accs = []
                for result_num in range(1, 6):
                    model_accs.append(results[f"{model_type}_{result_num}_similarity_{results_type}"][str(topics)])

                acc_avgs[f"{model_type}_{results_type}"].append(np.mean(model_accs))

            plt.plot(np.arange(MIN_TOPICS, MAX_TOPICS+1), acc_avgs[f"{model_type}_{results_type}"], label=f"{model_type}_{results_type}")

    plt.title("Top 5 Accuracy")
    plt.xlabel("# of Topics")
    plt.ylabel("Top 5 Acc%")
    plt.legend(loc="lower left", prop={'size': 8})
    plt.savefig(os.path.join(MAIN_PATH, "viz", "Top_5.png"))

    #######################################
    # 3. Correct% - Same% = Uniqueness
    #######################################
    plt.figure()
    plt.plot(np.arange(MIN_TOPICS, MAX_TOPICS+1), np.array(acc_avgs["lda_correct"]) - np.array(acc_avgs["lda_same"]), label="lda")
    plt.plot(np.arange(MIN_TOPICS, MAX_TOPICS+1), np.array(acc_avgs["lsi_correct"]) - np.array(acc_avgs["lsi_same"]), label="lsi")
    plt.title("Top 5 Uniqueness")
    plt.xlabel("# of Topics")
    plt.ylabel("Correct% - Same%")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(MAIN_PATH, "viz", "Uniqueness.png"))


def main():
    data = process_data.create_model_data()
    run_analysis(data)
    create_viz()


if __name__ == "__main__":
    main()
