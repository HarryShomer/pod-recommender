# Podcast Episode Recommendation Engine

## Steps to get the models

* To scrape all the data, run the `scrape_all.py` file in the scraper directory. This will scrape all the npr podcasts
and the This American Life podcast. It should take a little while to run.

* To clean and create the model (still in progress) run the `create_models.py` file in the models directory. This will
create and store an LDA and LSI model with 6 topics each. It will then generate the top 5 recommendations fot each podcast.
These will be stored in the `lsi_results.json` and `lda_results.json` files in the root directory. (NOTE: Be careful
running this as it will overwrite the graded results mentioned in the next section).

## Grading the model results

* Around ~10% of the results for both models were hand-graded. That is, using our own judgement we decided if each of
the 5 recommendations were either good or bad (to keep it simple we chose to keep the grading binary).

* We then compared the "accuracy" for each of the top 5 recommendations for each podcast. Running the file `check_results.py`
in the models directory will produce the graph `results.png` with the results. As seen the LDA model fares better than
the LSI and was chosen as our final model.

## Requirements:

Python 3.6+ is needed. The specific package requirements are listed in the requirements.txt file. To install all of them
in one shot just run the following command:

```
pip install -r requirements.txt
```
