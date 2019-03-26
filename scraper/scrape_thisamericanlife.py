from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import requests
import time
import os

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
fake_user = UserAgent(cache=True)


def scrape_episodes(episode_num):
    """
    Scrape the transcript from the page and store it.
    
    https://www.thisamericanlife.org/{episode_num}/transcript
    
    :param episode_num: Number episode in series
    
    :return: None
    """
    print(f"Scraping episode {episode_num}")

    response = requests.get(f"https://www.thisamericanlife.org/{episode_num}/transcript")
    soup = BeautifulSoup(response.content, "lxml")

    try:
        episode_title = soup.find("h1").text.replace(":", "-")
        episode_title = episode_title.replace("/", "-")
    except AttributeError:
        episode_title = f"Episode {episode_num}"

    # All the 'act's in the pod
    # Don't take last one because it is always the credits
    parts = soup.find_all("div", {"class": "act"})[:-1]

    # Get all the paragraphs
    paragraphs = []
    for part in parts:
        for paragraph in part.find_all("p"):
            paragraphs.append(paragraph.text)
    txt = " ".join(paragraphs)

    if len(txt) > 0:
        with open(os.path.join(FILE_DIR, "..", "podcast_data", "assorted", "this-american-life", episode_title + ".txt"), "w") \
                as file:
            file.write(txt)


def scrape_all():
    """
    Scrape all the episodes. There are currently 670 episodes. I just hardcoded it. 
    
    :return: None
    """
    # Dir to store transcripts
    if not os.path.isdir(os.path.join(FILE_DIR, "..", "podcast_data", "assorted", "this-american-life")):
        os.mkdir(os.path.join(FILE_DIR, "..", "podcast_data", "assorted", "this-american-life"))

    for num in range(1, 671):
        scrape_episodes(num)
        time.sleep(3)
