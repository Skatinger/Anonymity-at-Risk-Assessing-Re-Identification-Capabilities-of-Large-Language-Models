# convert raw wiki text to sentences, remove sentences which are too long

import nltk
import logging
from datasets import load_from_disk
import multiprocessing
from transformers import PegasusTokenizer

datasetPath = './data_reduced/'
logging.getLogger().setLevel(logging.INFO)
tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase', fast=True)


# use the SentencePiece model via nltk to split text into sentences
# https://arxiv.org/pdf/1808.06226.pdf
def split_to_sentences(example) -> dict:
    example['sentences'] = nltk.tokenize.sent_tokenize(example['text'], language='english')
    return example


# removes sentences from pages which have too many tokens after tokenization
# as too many tokens usually means that the sentence is either a faulty split and a full section,
# or contains special characters which contain no value for processing
def remove_long_sentences(example) -> dict:
    # truncate sentences longer than 256 tokens
    example['sentences'] = list(filter(lambda x: len(tokenizer(x)['input_ids']) <= 256, example['sentences']))
    return example


# loads the wikipedia dataset from huggingface if it does not yet exist
def load_wiki_dataset():
    logging.info('Loading dataset...')
    try:
        return load_from_disk(datasetPath)
    except ValueError as err:
        logging.warning("Specified dataset at ./data not available")
        logging.warning(err)
        quit()


if __name__ == '__main__':
    # read in the wiki-dataset
    dataset = load_wiki_dataset()

    # ensure required nltk package is installed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info("Downloading nltk punkt tokenizer")
        nltk.download('punkt')

    # use as many processes as possible
    cpus = multiprocessing.cpu_count()
    # don't use more than 8 cores, parallelization overhead will decrease performance
    cpus = min(cpus, 8)
    dataset = dataset.map(split_to_sentences, num_proc=cpus)

    # ditch sentences that are too long as they are often not useful
    dataset = dataset.map(remove_long_sentences, num_proc=cpus)

    # save the dataset to disk
    folder = './data_unparaphrased'
    dataset.save_to_disk(folder)

    logging.info("Saved unparaphrased wiki-dataset to {}".format(folder))
