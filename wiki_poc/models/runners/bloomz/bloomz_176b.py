import torch
from transformers import BloomTokenizerFast 
from petals import DistributedBloomForCausalLM
import os
import sys
from tqdm import tqdm
import signal
import logging
from datasets import load_dataset, Dataset

logging.getLogger().setLevel(logging.INFO)

def signal_handler(_sig, _frame):
    """allow signal handling, required when script is interrupted either with
        ctrl+c or with a proper sigint by the job handling server. All processing is cached to disk,
        this is just to ensure the job exits with a clean exit code and writes a short log to
        more easily see the reason for the exit upon log inspection.
    """
    logging.info("Received SIGINT, exiting.")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":

    # use CUDA if available (ususally has ID 0), -1 is huggingface default for CPU
    MODEL_NAME = "bigscience/bloomz-petals"

    CONFIG = "paraphrased"
    MODEL_NAME_SHORT = "bloomz-176b"
    PATH = f"wiki_predictions_{MODEL_NAME_SHORT}_{CONFIG}.jsonl"

    # dataset with initial pages
    dataset = load_dataset('Skatinger/wikipedia-persons-masked', CONFIG, split='train')
    # get set of page ids which are in the test_set_ids.csv file
    test_set_ids = set([i.strip() for i in open("test_set_ids.csv").readlines()])
    # filter out pages from dataset which are not in the test set
    dataset = dataset.filter(lambda x: x["id"] in test_set_ids)

    # only process pages which have not been processed yet, load processed pages from jsonl file if exists
    if os.path.exists(PATH):
        # store already processed results in result_dataset
        result_dataset = Dataset.from_json(PATH)
        # get set of page ids which have already been processed
        processed_ids = set(result_dataset['page_id'])
        # filter out pages from dataset which have already been processed
        dataset = dataset.filter(lambda x: x["id"] not in processed_ids)
    else:
        result_dataset = Dataset.from_dict({'prediction': [], 'page_id': [], 'input_length': []})

    # load model
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME)
    model.cuda()

    # prompts
    start_prompt = "The following text talks about a person but the person is referred to as <mask>.\n\n"
    end_prompt = "\n\nThe name of the person in the text referred to as <mask> is: "

    # create prediction for all pages in dataset and save them to a csv file
    for i, page in enumerate(tqdm(dataset)):
        # extract text from page
        text = page[f"masked_text_{CONFIG}"][:1000]
        # add start and end prompt
        prompt = start_prompt + text + end_prompt
        prompt_length = len(prompt)
        # tokenize text
        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        # generate prediction
        outputs = model.generate(inputs, max_new_tokens=5)
        # decode output
        prediction = tokenizer.decode(outputs[0])
        # remove prompts from prediction
        prediction = prediction[prompt_length:]
        # add prediction to result dataset
        result_dataset = result_dataset.add_item(
            {
                'prediction': prediction,
                'page_id': page['id'],
                'input_length': len(text)
             }
        )

        # periodically save file
        if i % 100 == 0:
            logging.info('Saving dataset intermediately to path %s', PATH)
            result_dataset.to_json(PATH)
        
    # save final dataset to file
    logging.info('Saving final dataset to path %s', PATH)
    result_dataset.to_json(PATH)
