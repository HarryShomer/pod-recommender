from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import requests
import time
import os

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, ElementNotVisibleException, WebDriverException
opts = Options()


PODCASTS =[
        "https://www.npr.org/podcasts/510310/npr-politics-podcast",
        "https://www.npr.org/series/423302056/hidden-brain",
        "https://www.npr.org/podcasts/510325/the-indicator-from-planet-money",
        "https://www.npr.org/podcasts/510321/wow-in-the-world",
        "https://www.npr.org/podcasts/510311/embedded",
        "https://www.npr.org/podcasts/510307/invisibilia",
        "https://www.npr.org/podcasts/510312/codeswitch",
        "https://www.npr.org/podcasts/510289/planet-money",
]


def get_transcript(url):
    """
    Get the transcript for the podcast episode

    :param url: String of url for transcript

    :return String of text in podcast
    """
    doc = requests.get(url).content
    doc = BeautifulSoup(doc, "lxml")
    paragraphs = doc.find("div", {"class": "transcript storytext"}).find_all("p")

    transcript_ps = []
    for paragraph in paragraphs:
        # Control for disclaimer at end
        if not paragraph.has_attr("class"):
            transcript_ps.append(paragraph.text)

    return " ".join(transcript_ps)


def parse_podcasts_episodes():
    """
    Scrape the transcript for all the podcasts where it exists
    """
    main_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "podcast_data", "npr")

    for pod in PODCASTS:
        # Get saved file of all episodes
        pod = pod[pod.rfind("/")+1:] 
        with open(os.path.join(main_dir, pod + ".html")) as file:
            pod_doc = file.read()

        print("\nScraping the episodes for", pod)

        # Create directory to hold transcripts for episode if not exist
        pod_dir = os.path.join(main_dir, pod)
        if not os.path.isdir(pod_dir):
            os.mkdir(pod_dir)

        # Parse Html
        pod_soup = BeautifulSoup(pod_doc, "lxml")
        classes = ["item podcast-episode", "item has-image has-audio", "item no-image has-audio"]
        episodes = pod_soup.find_all("article", {"class": classes})

        for episode in episodes:
            title = episode.find("h2", {"class": "title"}).text
            transcript_url = episode.find("li", {"class": "audio-tool audio-tool-transcript"})

            # Not all episodes have transcripts
            if transcript_url:
                print("Getting the transcript for", title)

                title = title.replace("/", "-")
                transcript = get_transcript(transcript_url.find("a")['href'])

                with open(os.path.join(pod_dir, title + ".txt"), "w") as file:
                    file.write(transcript)

                time.sleep(1)


def get_all_podcasts():
    """
    Get all the podcast episode info for the npr podcasts. Allows us
    to later scrape all the transcripts.

    :return None
    """
    # Create main dir if needed
    main_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "podcast_data", "npr")
    if not os.path.isdir(main_dir):
        os.mkdir(main_dir)

    fake_user = UserAgent(cache=True)

    for podcast in PODCASTS:
        print("Scraping", podcast)
        opts.add_argument(fake_user.random)

        # Get given url and html
        # To deal with dynamic content
        browser = webdriver.Chrome(chrome_options=opts)
        browser.get(podcast)

        # Wait up to 10 seconds for button
        wait = WebDriverWait(browser, 10)
        
        while True:
            try:
                wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'options__load-more')));
                browser.find_element_by_class_name('options__load-more').click()
                time.sleep(1)
            except (TimeoutException, ElementNotVisibleException, WebDriverException) as e:
                print("Finished clicking the button for this podcast")
                break

        file_dir = os.path.join(main_dir, podcast[podcast.rfind("/")+1:] + ".html")
        with open(file_dir, 'w') as file:
            file.write(browser.page_source)

        print("Sleeping for 10 seconds\n")
        time.sleep(10)

        browser.close()


def main():
    get_all_podcasts()
    parse_podcasts_episodes()

    
if __name__ == "__main__":
    main()