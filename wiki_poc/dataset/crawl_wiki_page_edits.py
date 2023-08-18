
"""script to download all the page edits for the wiki pages in the dataset

    the page edits are saved to dataset/wiki_page_edits.csv

    Data is collected from the Wikimedia REST API
    https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews
"""

import requests
import time
from tqdm import tqdm
import logging

from models.model_runner import load_test_set

def get_page_edits(titles):
    """returns the page views for the given titles"""
    page_edits = {}
    for title in tqdm(titles):
        page_edits[title] = get_page_edit(title)
        # slow down the requests a bit
        time.sleep(1)
    return page_edits

def get_page_edit(title):
    """returns the page view for the given title"""
                                    #  /api/rest_v1/metrics/edits/per-page/en.wikipedia/Barack+Obama/user/daily/20170101/20170201 
    BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/edits/per-page/en.wikipedia/"
    url = BASE_URL + title + "/user/monthly/2020010100/2020123100"
    headers = {
        # set the user agent to chrome
        'User-Agent': 'Fetching 10k page edits for the wiki dataset for master thesis'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return parse_page_edit(response.json())
    else:
        # try again in 30 seconds
        time.sleep(30)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return parse_page_edit(response.json())
        else:
            return -1

def parse_page_edit(json):
    """parses the page view json response"""
    if not json.get("items"):
        print("Items not found")
        return None
    count = 0
    for item in json["items"]:
        for result in item["results"]:
            count += result["edits"]
    return count

def main():
    # load the wiki dataset
    wiki_dataset = load_test_set(dataset_type='wiki')
    
    # get all the titles
    titles =  wiki_dataset['title']

    # get all the page views for the titles
    page_edits = get_page_edits(titles)

    # save a csv with only the page id and the page views
    with open("dataset/wiki_page_edits.csv", "w") as f:
        f.write("page_id,page_edits\n")
        for page, (_title, edits) in zip(wiki_dataset, page_edits.items()):
            f.write(f"{page['id']},{edits}\n")
    logging.info("saved page edits to dataset/wiki_page_edits.csv")

if __name__ == "__main__":
    main()