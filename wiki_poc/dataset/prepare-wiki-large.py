# helper script to download and save large wikipedia dataset on ubelix
# pulls a list of 700'000 people from wikidata, and only keeps wikipedia
# articles from this list of people.
# if an errors occurs in json.decoder, it's most likely that the wikipedia
# api had a timeout and rendered a java stacktrace into the response. Just
# rerun the script, or reduce the number of persons to query from wikipedia.
# If the error persists, check the last 500 characters of the response
# in `query_wiki_persons()` for the error message.

import logging
from datasets import load_dataset
from custom.wiki import query_wiki_persons
logging.getLogger().setLevel(logging.INFO)


# loads the wikipedia dataset from huggingface if it does not yet exist
def load_wiki_dataset():
    logging.info('Loading dataset...')
    try:
        return load_dataset("wikipedia", "20220301.en", split="train")
    except ValueError as err:
        logging.warning("Specified dataset not available, choose a current dataset:")
        logging.warning(err)
        quit()


if __name__ == '__main__':
    # load hugginface wikipedia dataset
    dataset = load_wiki_dataset()
    # load all titles of wiki pages which belong to a person
    logging.info("querying wiki database")
    persons = query_wiki_persons(700000)
    # create a set for faster processing
    persons = set(persons)
    # filter database to only keep rows which concern persons
    logging.info("filtering wiki database for persons")
    filteredDataset = dataset.filter(lambda example: example['title'] in persons)
    # save to disk for later usage
    logging.info("storing database to ./data/")
    filteredDataset.save_to_disk("./data/")
