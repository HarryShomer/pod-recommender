"""
Check the hand-graded results of the LDA and LSI models. Done on about ~10% of the results
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

MAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def check_model_results(model_type, topics):
    """
    Check the graded results of the model
    
    :param model_type: LDA or LSI
    :param topics: # of topics used for model
    
    :return: None
    """
    with open(os.path.join(MAIN_PATH, "..", "graded_results", f"{model_type.lower()}_{topics}_results.json"), "r") as f:
        results = json.load(f)['results']

    # Only keep results that are graded
    pods_graded = []
    graded_results = []
    for result in results:
        if 'Recommendation #1' in result and result['Recommendation #1']['isCorrect'] is not None:
            graded_results.append(result)
            pods_graded.append(result['Podcast'])

    # Add up recs by rec #
    recs = {f'Recommendation #{i}': [] for i in range(1, 6)}
    for result in graded_results:
        for rec in recs:
            recs[rec].append(result[rec]['isCorrect'])

    return [sum(recs[f'Recommendation #{i}']) / len(recs[f'Recommendation #{i}']) for i in range(1, 6)]


def plot_lda_results():
    """
    Plot the results for the 3 LDA models (topic of 5, 6, and 7)
    """
    plt.figure()

    index = np.arange(5)
    bar_width = .3
    N = 3  # num models

    # Create axis for each
    lda_models = []
    for i, topics in zip(range(0, N), [5, 6, 7]):
        width_offset = bar_width * i
        lda_models.append(plt.bar(index + width_offset, check_model_results("LDA", topics), bar_width,
                                  label=f"lda_{topics}"))

    plt.xlabel("Rec #")
    plt.ylabel("Accuracy")
    plt.xticks(index + bar_width, index + 1)

    plt.title(f"LDA Model Accuracy")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(MAIN_PATH, "..", f"lda_results.png"))


def plot_initial_results():
    """
    Plot Results of both initial models.
    
    This is LDA vs. LSI for 6 topics
    """
    plt.figure()

    index = np.arange(5)
    width = .3

    plt.bar(index, check_model_results("LSI", 6), width, label=f"LSI")
    plt.bar(index + width, check_model_results("LDA", 6), width, label=f"LDA")

    plt.xlabel("Rec #")
    plt.ylabel("Accuracy")
    plt.xticks(index + width / 2, index+1)

    plt.title(f"LDA vs. LSI Accuracy Using 6 Topics")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(MAIN_PATH, "..", f"lda_vs_lsi_results.png"))


if __name__ == "__main__":
    plot_initial_results()
    plot_lda_results()
