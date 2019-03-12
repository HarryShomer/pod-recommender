# Podcast Episode Recommendation Engine

NOTE: Ignore everything in the models directory as it is out of date.

## Steps to get the models

* First scrape all the raw data run the scrape_itunes.py file. This will give us everything we need to do this. It can
take a while to run (can also just download from [here](https://www.dropbox.com/s/17sj74s543h1e6g/all_pod_episodes.csv?dl=0) and place it in the root directory of the project). 

* Once you have that data you can run the process_data.py file in the models directory to transform the raw data to what we will give the model. I recommend looking at the file to see what I did and offer suggestions (you should probably familiarize yourself with the data as well). This should take about 15 minutes to run.

* TODO: Create the model

## Requirements:

Python 3.6+ is needed. The specific package requirements are listed in the requirements.txt file. To install all of them
in one shot just run the following command:

```
pip install -r requirements.txt
```
