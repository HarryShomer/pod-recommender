# Podcast Episode Recommendation Engine

## Steps to get the models

* To scrape all the data, run the `scrape_all.py` file in the scraper directory. This will scrape all the npr podcasts
and the This American Life podcast. It should take a little while to run.

* To clean and create the model (still in progress) run the `create_models.py` file in the models directory. This will
create and store all the models along with the recommendations produced for each podcast by model.

## Requirements:

Python 3.6+ is needed. The specific package requirements are listed in the requirements.txt file. To install all of them
in one shot just run the following command:

```
pip install -r requirements.txt
```
