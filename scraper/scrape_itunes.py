import requests
from bs4 import BeautifulSoup
import pandas as pd
from fake_useragent import UserAgent
import json
import os
import time


def scrape_genre_pg(url):
    """
    Scrape the page for the genre. Contains names of top 240 podcasts

    :param url: Url of genre page
    
    :return: List of general podcast info
    """
    fake_user = UserAgent(cache=True)
    response = requests.get(url, headers={'User-Agent': fake_user.random})
    pg_soup = BeautifulSoup(response.content, "lxml")

    podcast_info = []

    for col in ['column first', 'column', 'column last']:
        podcasts = pg_soup.find("div", {"class": col}).find_all("a")
        for podcast in podcasts:
            pod = ({
                "name": podcast.text.replace("/", "-"),            # Forwardlashes in podcast name fucks with filepaths
                "url": podcast['href'].replace("itmss", "https")   # Sometimes it's for Itunes app
            })

            # Run into issue with duplicates
            if pod not in podcast_info:
                podcast_info.append(pod)

    return podcast_info


def scrape_genre_pgs():
    """
    Scrape the info for all the genre pages and store in a json file.
    Store name and url for each podcast by genre
    
    :return: None 
    """
    base_url = "https://itunes.apple.com/us/genre/podcasts-kids-family/id1305?mt=2"
    pg_soup = BeautifulSoup(requests.get(base_url).content, "lxml")

    # Find list of genres
    genre_nav = pg_soup.find_all("div", {'id': "genre-nav"})
    genre_list = genre_nav[0].find_all("li")

    total_podcasts = {}
    for genre in genre_list:
        print("Scraping the page for", genre.text)
        genre_pods = scrape_genre_pg(genre.find("a")["href"])
        total_podcasts[genre.text] = genre_pods

    with open("all_podcasts.json", "w+") as file:
        json.dump(total_podcasts, file, indent=4)


def scrape_podcast_pg(pod_info, genre, fake_user, pod_episodes):
    """
    Scrape the podcast page, store the html, and return a list of the podcasts
    
    :param pod_info: Dict with "name" & "url"
    :param fake_user: Mask user_agent
    :param genre: Genre of the podcast
        
    :return: List of episodes for podcast
    """
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "podcast_docs",
                             genre, ".".join([pod_info['name'], "html"]))

    # Check if the file was already scraped
    # If it was load it in
    # Otherwise get it and save it for next time
    if not os.path.exists(file_path):
        doc = requests.get(pod_info['url'], headers={'User-Agent': fake_user.random}, timeout=5).content
        time.sleep(1)
        with open(file_path, 'wb') as file:
            file.write(doc)
        return
    else:
        with open(file_path, 'rb') as my_file:
            doc = my_file.read()

    pg_soup = BeautifulSoup(doc, "lxml")

    # General description of pod
    pod_description = pg_soup.find("div", {"metrics-loc": "Titledbox_Description"})

    try:
        pod_description = pod_description.find("p").text.strip()
    except AttributeError:
        print(pod_info['name'], "has no podcast description")
        return None

    # Get episodes for podcast
    # 1 onwards because first is head
    episode_trs = pg_soup.find("div", {"class": "tracklist-content-box"})
    episode_trs = episode_trs.find_all("tr")[1:]

    language = pg_soup.find("li", {"class": "language"}).text
    language = language[len("Language:") + 1:].strip()

    for episode in episode_trs:
        tds = episode.find_all("td")

        if language == "English":
            pod_episodes.append({
                "pod_name": pod_info['name'],
                "episode_name": tds[1]['sort-value'].strip(),
                "episode_description": tds[2]['sort-value'].strip(),
                "pod_description": pod_description,
                "release_date": tds[3]['sort-value'].strip(),
                "genre": genre
            })


def get_data():
    """
    Get all of the required data
    
    :return: None
    """
    file_dir = os.path.dirname(os.path.realpath(__file__))
    podcast_path = os.path.join(file_dir, "all_podcasts.json")

    # It is stored so only needs to be done once
    if not os.path.isfile(podcast_path):
        scrape_genre_pgs()

    with open(podcast_path) as file:
        all_podcasts = json.load(file)

        # Store scraped html in here
        if not os.path.isdir(os.path.join(file_dir, "podcast_docs")):
            os.mkdir(os.path.join(file_dir, "podcast_docs"))
            for genre in all_podcasts:
                os.mkdir(os.path.join(file_dir, "podcast_docs", genre))

    fake_user = UserAgent(cache=True)

    # Save data for all individual podcasts
    pod_episodes = []
    for genre in all_podcasts:
        print("Processing", genre)
        for pod in all_podcasts[genre]:
            scrape_podcast_pg(pod, genre, fake_user, pod_episodes)

    df = pd.DataFrame(pod_episodes)
    df.to_csv("../all_pod_episodes.csv", index=False, sep=',')


if __name__ == "__main__":
    get_data()

