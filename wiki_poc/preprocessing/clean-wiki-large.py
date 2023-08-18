# for a good base dataset, remove outliers

# removes:
# - the "links" section of pages
# - the "references" section of pages
# - pages with less than 4'000 characters (approx 800 words => 1 batch ~ 500 words)

import logging
from datasets import load_from_disk
import re

# sigint handler
import signal
import sys


# allow signal handling
def signal_handler(sig, frame):
    logging.info("Received SIGINT, saving checkpoint")
    global dataset
    dataset.save_to_disk("./data")
    logging.info("exiting")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)

logging.getLogger().setLevel(logging.INFO)


# loads the wikipedia dataset from huggingface if it does not yet exist
def load_wiki_dataset():
    logging.info('Loading dataset...')
    try:
        return load_from_disk("../dataset/data")
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
    dataset = dataset.map(remove_links, batched=True, num_proc=4)
    # only keep pages with more than 6'000 characters
    dataset = dataset.filter(lambda x: len(x['text']) > 6000)
    # save the dataset
    dataset.save_to_disk("../dataset/data")
