# for a good base dataset, remove outliers

# removes:
# - the "links" section of pages
# - the "references" section of pages
# - pages with less than 6'000 characters

import logging
from datasets import load_from_disk
import re

logging.getLogger().setLevel(logging.INFO)


# loads the wikipedia dataset from huggingface if it does not yet exist
def load_wiki_dataset():
    logging.info('Loading dataset...')
    try:
        return load_from_disk("./data")
    except ValueError as err:
        logging.warning("Specified dataset at ./data not available")
        logging.warning(err)
        quit()


# helpers for mapping over the dataset
def remove_links(page):
    match = re.match('Bibliography\n\n', page['text'])
    if match:
        page['text'] = page['text'][:match.start()]
    match = re.match('References\n\n', page['text'])
    if match:
        page['text'] = page['text'][:match.start()]
    return page


if __name__ == '__main__':
    # load hugginface wikipedia dataset
    dataset = load_wiki_dataset()
    # remove the "links", "bibliography" and "references" section of pages
    logging.info("Removing links, bibliography and references")
    dataset = dataset.map(remove_links, num_proc=8)
    # only keep pages with more than 6'000 characters
    logging.info("Removing pages with less than 6'000 characters")
    dataset = dataset.filter(lambda x: len(x['text']) > 6000)
    # save the dataset
    logging.info("Saving dataset to disk")
    dataset.save_to_disk("./data_reduced")
