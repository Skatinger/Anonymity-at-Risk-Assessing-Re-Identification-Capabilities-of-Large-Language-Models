
"""script to download all the page views for the wiki pages in the dataset

    the page views are saved to dataset/wiki_page_views.csv

    Data is collected from the Wikimedia REST API
    https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews
"""

import requests
import time
from tqdm import tqdm
import logging

from models.model_runner import load_test_set

def get_page_views(titles):
    """returns the page views for the given titles"""
    page_views = {}
    for title in tqdm(titles):
        page_views[title] = get_page_view(title)
        # slow down the requests a bit
        time.sleep(0.01)
    return page_views

def get_page_view(title):
    """returns the page view for the given title"""
    BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/"
    url = BASE_URL + title + "/monthly/2020010100/2020123100"
    headers = {
        # set the user agent to chrome
        'User-Agent': 'Fetching 10k page views for the wiki dataset for master thesis'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return parse_page_view(response.json())
    else:
        return None

def parse_page_view(json):
    """parses the page view json response"""
    if not json.get("items"):
        return None
    return sum([item["views"] for item in json["items"]])

def main():
    # load the wiki dataset
    wiki_dataset = load_test_set(dataset_type='wiki')
    
    # get all the titles, replace spaces with underscores to match the api naming convention
    titles = [title.replace(" ", "_") for title in wiki_dataset['title']]

    # get all the page views for the titles
    page_views = get_page_views(titles)

    # save a csv with only the page id and the page views
    with open("dataset/wiki_page_views.csv", "w") as f:
        f.write("page_id,page_views\n")
        for page, (_title, views) in zip(wiki_dataset, page_views.items()):
            f.write(f"{page['id']},{views}\n")
    logging.info("saved page views to dataset/wiki_page_views.csv")

if __name__ == "__main__":
    main()