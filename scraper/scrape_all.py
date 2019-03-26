"""
Scrape all podcasts
"""
import scrape_npr
import scrape_thisamericanlife

scrape_npr.get_all_podcasts()
scrape_npr.parse_podcasts_episodes()
scrape_thisamericanlife.scrape_all()
