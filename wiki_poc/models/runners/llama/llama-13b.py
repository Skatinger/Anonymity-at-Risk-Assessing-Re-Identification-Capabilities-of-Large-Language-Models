import os
from datasets import load_dataset
from datasets import Dataset
import logging
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

"""
This script is used to test facebooks llama model on the wikipedia dataset.
Mask-Filling is not possible, so we use prompt completion to prompt for the name referred to as <mask>.

TODO: check where the cutoff is for the model to recognize the person, e.g. how many characters are needed and
at what point is it useless to send any more characters to the model. This improves cost efficiency.
"""

if __name__ == "__main__":

    # ensure required test ids file exists
    assert os.path.exists("test_set_ids.csv"), "test_set_ids.csv file not found. Please run generate_test_set_ids.py first."
    # assert that PATH_TO_CONVERTED_WEIGHTS  and PATH_TO_CONVERTED_TOKENIZER are set
    assert os.getenv("PATH_TO_CONVERTED_WEIGHTS") is not None, "PATH_TO_CONVERTED_WEIGHTS environment variable not set."
    assert os.getenv("PATH_TO_CONVERTED_TOKENIZER") is not None, "PATH_TO_CONVERTED_TOKENIZER environment variable not set."

    CONFIG = "paraphrased"
    MODEL_NAME = "decapoda-research/llama-65b-hf"
    PATH = f"wiki_predictions_{MODEL_NAME.replace('/', '_')}_{CONFIG}.jsonl"
    PATH_TO_CONVERTED_WEIGHTS = os.getenv("PATH_TO_CONVERTED_WEIGHTS")
    PATH_TO_CONVERTED_TOKENIZER = os.getenv("PATH_TO_CONVERTED_TOKENIZER")


    start_prompt = "The following text refers to a person with <mask>:\n"
    end_prompt = "\nThe exact name of that person is:"

    # dataset with initial pages
    dataset = load_dataset('Skatinger/wikipedia-persons-masked', CONFIG, split='train')
    # get set of page ids which are in the test_set_ids.csv file
    test_set_ids = set([i.strip() for i in open("test_set_ids.csv").readlines()])
    # filter out pages from dataset which are not in the test set
    dataset = dataset.filter(lambda x: x["id"] in test_set_ids)

    # only process pages which have not been processed yet
    if os.path.exists(PATH):
        # store already processed results in result_dataset
        result_dataset = Dataset.from_json(PATH)
        # get set of page ids which have already been processed
        processed_ids = set(result_dataset['page_id'])
        # filter out pages from dataset which have already been processed
        dataset = dataset.filter(lambda x: x["id"] not in processed_ids)
    
    else:
        result_dataset = Dataset.from_dict({'prediction': [], 'page_id': [], 'input_length': []})

    # load model and tokenizer
    model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    # iterate over pages in dataset
    for index, page in enumerate(tqdm(dataset)):

        # extract text from page
        text = page[f"masked_text_{CONFIG}"][:1000]

        # prompt model for output
        prompt = start_prompt + text + end_prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate
        generate_ids = model.generate(inputs.input_ids, max_length=30)
        result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


        # add prediction to result dataset
        result_dataset = result_dataset.add_item(
            {'prediction': result,
             'page_id': page['id'], 'input_length': len(text)})
    
        # periodically save file
        if index % 100 == 0:
            logging.info('Saving dataset intermediately to path %s', PATH)
            result_dataset.to_json(PATH)

    # save dataset
    logging.info('Saving dataset to path %s', PATH)
    result_dataset.to_json(PATH)