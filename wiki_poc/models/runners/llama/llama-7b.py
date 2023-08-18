import os
import sys
import torch
import signal
import logging
from transformers import LlamaForCausalLM, LlamaTokenizer
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

global CONFIG
global DEVICE
global tokenizer
global model_8bit

# run predictions with model.generate instead of pipeline to save memory and speed up processing
def run_prediction(examples):
    # tokenize inputs and move to GPU
    inputs = tokenizer(examples[f"masked_text_{CONFIG}"], return_tensors="pt", padding=True).to(DEVICE)
    # generate predictions
    generated_ids = model_8bit.generate(**inputs, early_stopping=True, num_return_sequences=1, max_new_tokens=5)
    # decode predictions
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # get prediction and remove the input from the output
    predictions = [out.replace(examples[f"masked_text_{CONFIG}"][i], "") for i, out in enumerate(outputs)]
    input_lengths = [len(i) for i in examples[f"masked_text_{CONFIG}"]]
    return { "prediction": predictions, "page_id": examples["id"], "input_length": input_lengths }

if __name__ == "__main__":

    MODEL_NAME = "decapoda-research/llama-7b-hf"

    assert torch.cuda.is_available(), "CUDA is not available. Please install CUDA and try again."
    DEVICE = 0
    
    CONFIG = "paraphrased"
    MODEL_NAME_SHORT = "llama-7b"
    PATH = f"wiki_predictions_{MODEL_NAME_SHORT}_{CONFIG}.jsonl"

    # dataset with initial pages
    dataset = load_dataset('Skatinger/wikipedia-persons-masked', CONFIG, split='train')
    # get set of page ids which are in the test_set_ids.csv file
    test_set_ids = set([i.strip() for i in open("test_set_ids.csv").readlines()])
    # filter out pages from dataset which are not in the test set
    dataset = dataset.filter(lambda x: x["id"] in test_set_ids, num_proc=8)

    # only process pages which have not been processed yet, load processed pages from jsonl file if exists
    if os.path.exists(PATH):
        # store already processed results in result_dataset
        result_dataset = Dataset.from_json(PATH)
        # get set of page ids which have already been processed
        processed_ids = set(result_dataset['page_id'])
        # filter out pages from dataset which have already been processed
        dataset = dataset.filter(lambda x: x["id"] not in processed_ids)

    # load model
    MODEL_WEIGHTS_PATH = '/home/alex/.cache/llama-converted/'
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_WEIGHTS_PATH)
    tokenizer.pad_token = tokenizer.eos_token # define pad token as eos token
    model_8bit = LlamaForCausalLM.from_pretrained(MODEL_WEIGHTS_PATH, device_map="auto", load_in_8bit=True)

    # shorten text to 1000 characters
    dataset = dataset.map(lambda x: {f"masked_text_{CONFIG}": x[f"masked_text_{CONFIG}"][:1000]}, num_proc=8)

    # prompts
    start_prompt = "The following text talks about a person but the person is referred to as <mask>.\n\n"
    end_prompt = "\n\nThe name of the person in the text referred to as <mask> is: "

    # prepend start and end prompt to all examples
    dataset = dataset.map(lambda x: {f"masked_text_{CONFIG}": start_prompt + x[f"masked_text_{CONFIG}"] + end_prompt})

    result_dataset = dataset.map(run_prediction, batched=True, batch_size=64, remove_columns=dataset.column_names)

    # save final dataset to file
    logging.info('Saving final dataset to path %s', PATH)
    result_dataset.to_json(PATH)