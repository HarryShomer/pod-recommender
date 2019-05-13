# Podcast Episode Recommendation Engine

The work presented here is to create a model that will recommend 5 new podcast episodes to listen to given one 
episode. The data used for each episode is the full transcript. 

## Methodology

### Data Preparation
To achieve this we collected episode transcripts for episodes of 14 different podcasts. This came out to a total of 2028 different episodes. The transcript data was first cleaned (we attempted to get rid of the names of hosts and other words not relevant to the actual content of the episode) and tokenized. We then got rid of all punctuation and stop words, and converted every token to lowercase. Each token was then lemmatized. 

### Models and Grading
We then considered using to modeling techniques: latent dirichlet allocation (LDA) and latent semantic indexing (LSI). To this point we created a model using each technique using 6 topics (that is, we assumed the episodes could be divided into
6 topics). 

In order to determine which of the two models performed better we then hand-graded about 10% of the results for both models. That is, using our own judgement we decided if each of the 5 recommendations were either good or bad (to keep it simple we chose to keep the grading binary). These results are all stored in the `graded_results` directory. The results can be seen below:

![](https://github.com/HarryShomer/pod-recommender/blob/master/lda_vs_lsi_results.png)

These results indicate the LDA model performed better. We then tried to fit two new different LDA models, with 5 and 7 topics to see if they performed better(a reminder that the previous one was 6). We then hand-graded the same proportion of results as earlier and looked at the results. The LDA model with 6 topics still performed better. The results can be seen below:

![](https://github.com/HarryShomer/pod-recommender/blob/master/lda_results.png)


## Running it yourself

Before anything make sure you satisfy the requirements listed in the [requirements](#Requirements) section below. 

### Steps to get the models

* To scrape all the data, run the `~/scraper/scrape_all.py` file. This will scrape all the npr podcasts and the This American Life podcast. 

* To clean the data and create the models run the `~/models/create_models.py` file. This will create and store all the models used in this project. It will then generate the top 5 recommendations for each podcast. These will be stored in the appropriate folder in the models directory. For example, for the LDA model of 5 topics it will be stored in 
`~/models/lda/5_topics/lda_5_results.json`. 

* Lastly, to produce the two vizualizations of the results, you'll need to run the `~/models/check_results.py` file. This will generate both of them in the root directory. These vizualiations rely on the results we graded and stored in the 
`graded_results` directory. 


## Requirements:

Python 3.6+ is needed. The specific package requirements are listed in the requirements.txt file. To install all of them
in one shot just run the following command:

```
pip install -r requirements.txt
```

You may also need to install the Selenium WebDriver for your specific browser. They can be found [here](https://www.seleniumhq.org/download/). 