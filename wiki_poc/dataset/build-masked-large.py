# recognizes entities in the given sentences and masks them accordingly
# stores masked sentences and the tokens belonging to the masks

import faulthandler
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch
from datasets import load_from_disk, concatenate_datasets
import functools
import re

# sigint handler
import signal
import sys
import logging
logging.getLogger().setLevel(logging.INFO)


# allow signal handling, required when script is interrupted either with ctrl+c or with a proper sigint
# by the job handling server. All processing is cached to disk, this is just to ensure the job exits
# with a clean exit code and writes a short log to more easily see the reason for the exit upon log inspection
def signal_handler(sig, frame):
    logging.info("Received {}, exiting.".format(sig))
    sys.exit(0)


# signal.signal(signal.SIGTERM, signal_handler)

# ensure error stack is printed when an error occurs on the GPU / Computing Cluster
# faulthandler.enable()

# use CUDA if available, careful: usually the CUDA device ID is zero. If it is different
# on the system in use, change this number to the correct device numer.
device = 0 if torch.cuda.is_available() else -1

# define the input dataset, if the pipeline was not changed, the last step in processing the dataset
# should have automatically created this folder.
datasetPath = './data_paraphrased'


# load the pipeline to apply named entity recognition. Technically the huggingface library should
# automatically keep the model in memory and cache the loading form the web, but to be sure the lru_cache is used,
# ensuring the procedure is stored in memory.
@functools.lru_cache(maxsize=1)
def load_ner_pipeline(model_name="dslim/bert-base-NER"):
    print(f"Loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # push the model to the GPU if available
    logging.info("Using device: {}".format(device))
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    return pipeline("ner", model=model, tokenizer=tokenizer, device=device)


# takes the result of the bert-base-NER pipeline, the text which should be masked and the entity which should be masked
# returns the masked text and the entities which were masked, one entity in the entities array per mask in the text
# This works by
# 1. parsing the predictions of the NER pipeline.
# 2. Then building a regex which should match all versions of the name passed in as entity_to_mask (e.g. only firstname,
#    only lastname, middlename included or not and so on)
# 3. Then weeding out all predictions which do not match the regex (neglecting any other persons except the one we are
#    looking for)
# 4. Then masking the text by replacing the matched indices with the mask token. This is done in reverse, meaning it starts
#    processing at the end of the passed text and ends at the start. This ensures that indices are not shifted when
#    replacing text with the mask token, which would lead to shifted masks, with the shift getting increasingly worse
#    towards the end.
def masking(ner_results, text, entity_to_mask, batch_sizes, mask_token='<mask>'):
    person_nr = -1
    entities = []
    indexes = []
    index_start = 0
    for batch_size, result in zip(batch_sizes, ner_results):
        # reset last_end index for each batch, we do not want to concat entities from different batches, as batches
        # are separated by sentences (and thus entities cannot overlap)
        last_end = -10
        for entity in result:
            tag = entity['entity']
            if 'PER' in tag:  # if it is a person
                if ('B' in tag or 'I' in tag) and '#' not in entity['word']:
                    # if end of entity is right after the end of the last entity, merge them
                    if entity['start'] - 1 == last_end:
                        # concat to last entity part with empty space between
                        entities[person_nr] += ' ' + entity['word']
                        indexes[person_nr][1] = entity['end'] + index_start
                        last_end = entity['end'] + index_start
                    else:
                        person_nr += 1  # we came to the next person
                        entities.append(entity['word'].strip())
                        indexes.append([entity['start'] + index_start, entity['end'] + index_start])
                        last_end = entity['end']
                else:
                    if person_nr < 0:
                        continue
                    if '#' in entity['word']:
                        entities[person_nr] += entity['word'].strip().strip('#')
                        indexes[person_nr][1] = entity['end'] + index_start
                        last_end = entity['end'] + index_start
                    else:
                        entities[person_nr] += ' ' + entity['word']
                        indexes[person_nr][1] = entity['end'] + index_start
                        last_end = entity['end'] + index_start
        # update index_start for next batch, so indexes can be directly used for masking in full text
        # otherwise we would have to add the length of the previous batch to the indexes
        index_start += batch_size
    # remove entities which are not the ones we want to mask, e.g. remove persons which
    # are not the person the article is about
    # complex checker:
    regex = ".*".join(entity_to_mask.split(" "))
    regex += ".*".join(['|.*(' + nameFragment + ').*' for nameFragment in entity_to_mask.split(" ")])

    # weed out entities which do not match the regex which identifies the person we want to mask
    remaining_entities = []
    remaining_indexes = []
    for entity, index in zip(entities, indexes):
        if bool(re.match(regex, entity)):
            remaining_entities.append(entity)
            remaining_indexes.append(index)

    # return a dataset of anonymized strings with the belonging entities
    # https://stackoverflow.com/questions/69921629/transformers-autotokenizer-tokenize-introducing-extra-characters
    # escape entity to ensure no brackets or other regex keywords are present (would cause a parsing error), for example
    # 'Heinrich ) Frank' causes a unbalanced parenthesis error
    text = "".join(text)
    for index in reversed(remaining_indexes):
        # the index start of an entity changes when text before it is masked, so we mask in reverse order
        text = text[:index[0]] + mask_token + text[index[1]:]

    return text, remaining_entities


# loads the dataset containing original and paraphrased texts, exits in case it does not yet exist.
def load_wiki_dataset():
    logging.info('Loading dataset...')
    try:
        return load_from_disk(datasetPath)
    except ValueError as err:
        logging.warning("Specified dataset at {} not available".format(datasetPath))
        logging.warning(err)
        quit()


# helper to prepare the batched sentences passing the paraphrased type
def prepare_paraphrased(example):
    return prepare_batched(example, 'paraphrased', 'paraphrased_sentences')


# helper to prepare the batched sentences passing the original type
def prepare_original(example):
    return prepare_batched(example, 'original', 'sentences')


# runs the ner pipeline on the given example and masks the entities
# example is a wiki page, type is either 'sentences' or 'paraphrased_sentences', depending on which column to use
def prepare_batched(example, type, colname):
    # split sentences array in batches of splitSize 7
    splitSize = 7
    # [["sentence 1", "sentence 2", ...], [...]]
    sentencesBatches = [example[colname][i:i + splitSize] for i in range(0, len(example[colname]), splitSize)]

    # join sentences within each batch with spaces to get a single string to pass to NER pipeline
    sentencesBatches = [' '.join(sentences) for sentences in sentencesBatches]
    batchesLenths = [len(text) for text in sentencesBatches]
    return {'batched_{}_sentences'.format(type): sentencesBatches, 'batched_{}_sizes'.format(type): batchesLenths}


def apply_original_ner(example):
    return apply_ner(example, 'original')


def apply_paraphrased_ner(example):
    return apply_ner(example, 'paraphrased')


# performs ner on the given example on the given column
def apply_ner(example, type):
    ner = load_ner_pipeline()
    column = 'batched_{}_sentences'.format(type)
    ner_results = ner(example[column])
    return {"ner_results_{}".format(type): ner_results}


def apply_original_masking(example):
    return apply_masking(example, 'original')


def apply_paraphrased_masking(example):
    return apply_masking(example, 'paraphrased')


# masks the entities in the given example on the given column
def apply_masking(example, type):
    # mask entities in the text
    ner_results = example['ner_results_{}'.format(type)]
    batched_text = example['batched_{}_sentences'.format(type)]
    batch_sizes = example['batched_{}_sizes'.format(type)]
    masked_text, entities = masking(ner_results, batched_text, example['title'], batch_sizes)

    return {'masked_text_{}'.format(type): masked_text, 'masked_entities_{}'.format(type): entities}


if __name__ == '__main__':
    # read in the wiki-dataset
    dataset = load_wiki_dataset()

    # number of shard splits to create when processing map function for full dataset takes a long time
    # each shard gets cached seperately, so we can process the dataset in multiple runs without complicated
    # cancellation or error handling
    numShards = 70

    # split sentences of each page into batches of 7 sentences, and sentences
    # within each batch into a single string for better performance and accuracy
    # no need to shard this step, as it is comparatively fast
    # as this is mainly a simple datastructure manipulation step, it runs on CPU, use more
    # cores to speed up processing. Core number is automatically reduced if machine has less cores
    logging.info('Splitting original sentences into batches...')
    dataset = dataset.map(prepare_original, num_proc=16)
    logging.info('Splitting paraphrased sentences into batches...')
    dataset = dataset.map(prepare_paraphrased, num_proc=16)

    # apply ner pipeline to each batch of sentences
    # do it sharded to allow caching each shard, in case of a crash
    logging.info('Applying NER pipeline...')
    computedOriginalShards = []
    for shardIndex in range(0, numShards):
        logging.info('Processing shard {}/{}'.format(shardIndex, numShards))
        # no paralellization here, as the bottleneck is the GPU, not the CPU
        computedOriginalShards.append(dataset.shard(numShards, shardIndex).map(apply_original_ner))
    dataset = concatenate_datasets(computedOriginalShards)

    computedParaphrasedShards = []
    for shardIndex in range(0, numShards):
        computedParaphrasedShards.append(dataset.shard(numShards, shardIndex).map(apply_paraphrased_ner))
    dataset = concatenate_datasets(computedParaphrasedShards)

    # mask entities detected in NER
    logging.info('Masking entities...')
    # use more CPUs as this is a CPU only operation, primarily running regex on texts
    dataset = dataset.map(apply_original_masking, num_proc=16)
    dataset = dataset.map(apply_paraphrased_masking, num_proc=16)

    # drop rows where we got no entities, as they are not interesting for our project
    logging.info('Dropping rows without entities...')
    dataset = dataset.filter(lambda example: len(example['masked_entities_original']) > 0, num_proc=16)
    dataset = dataset.filter(lambda example: len(example['masked_entities_paraphrased']) > 0, num_proc=16)

    # remove columns we don't need anymore
    logging.info('Removing unnecessary columns...')
    cols = ['batched_original_sentences', 'batched_original_sizes', 'batched_paraphrased_sentences',
            'batched_paraphrased_sizes', 'ner_results_original', 'ner_results_paraphrased']
    dataset = dataset.remove_columns(cols)

    # save the dataset
    logging.info('Saving dataset to ./data_masked...')
    dataset.save_to_disk('./data_masked')
