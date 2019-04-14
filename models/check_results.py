"""
Check the hand-graded results of the LDA and LSI models. Done on about ~10% of the results
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

MAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def check_model_results(model_type):
    """
    Check the graded results of the model
    
    :param model_type: LDA or LSI
    
    :return: None
    """
    with open(os.path.join(MAIN_PATH, f"{model_type.upper()}.json"), "r") as f:
        results = json.load(f)['results']

    # Only keep results that are graded
    graded_results = []
    for result in results:
        if 'Recommendation #1' in result and result['Recommendation #1']['isCorrect'] is not None:
            graded_results.append(result)

    # Add up recs by rec #
    recs = {f'Recommendation #{i}': [] for i in range(1, 6)}
    for result in graded_results:
        for rec in recs:
            recs[rec].append(result[rec]['isCorrect'])

    return [sum(recs[f'Recommendation #{i}']) / len(recs[f'Recommendation #{i}']) for i in range(1, 6)]


def plot_results():
    """
    Plot Results of both models
    """
    plt.figure()

    index = np.arange(5)
    width = .3

    plt.bar(index, check_model_results("LSI"), width, label=f"LSI")
    plt.bar(index + width, check_model_results("LDA"), width, label=f"LDA")

    plt.xlabel("Rec #")
    plt.ylabel("Accuracy")
    plt.xticks(index + width / 2, index+1)

    plt.title(f"LDA vs. LSI Accuracy")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(MAIN_PATH, f"results.png"))


if __name__ == "__main__":
    plot_results()
