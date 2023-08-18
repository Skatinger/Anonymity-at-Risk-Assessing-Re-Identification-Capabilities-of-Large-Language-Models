
from .abstract_runner import AbstractRunner
from transformers import AutoModelForQuestionAnswering
import logging
import torch
from transformers import pipeline
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset

class AbstractQARunner(AbstractRunner):

    def _model_loader(self):
        return AutoModelForQuestionAnswering


    def start_prompt(self):
        """returns the start prompt for the model"""
        return "What is the exact name of the person referred to as <mask>? Hint: The answer is NOT <mask>."

    def prepare_examples(self):
        """shortens input text to max length given and prepend question to context"""
        logging.info(f"Preparing examples for {self.model_name}")
        self.examples = {}
        for config in self.configs:
            # shorten input text to max length given
            df = self.dataset.map(lambda x: {f"masked_text_{config}": x[f"masked_text_{config}"][:self.input_length]}, num_proc=8)
            # remove all examples which do no longer contain a mask
            df = df.filter(lambda x: '<mask>' in x[f"masked_text_{config}"], num_proc=8)
            # format examples for QA as dict with question and context
            df = df.map(lambda x: {f"qa_{config}": {"question": self.start_prompt(), "context": x[f"masked_text_{config}"]}})
            self.examples[config] = df

    def load_pipe(self):
        logging.info(f"Loading pipeline for {self.model_name}")
        if not torch.cuda.is_available():
            logging.warning("GPU not available, loading pipeline in FP32 mode on CPU. This will be very slow.")
        # pipeline is automatically loaded on GPU if available when loading the model in 8bit mode
        return pipeline('question-answering', model=self.model, tokenizer=self.tokenizer, top_k=self.k_runs)

    def run_model(self):
        # check if results already exist
        cached = self.check_cache()
        if all(cached.values()):
            logging.info(f"Results already exist, skipping model {self.model_name}")
            return
        # load tokenizer
        self.tokenizer = self.get_tokenizer()
        # prepare examples for different configs
        self.prepare_examples()
        # load tokenizer and model
        self.model = self.get_model()
        # load pipeline
        pipe = self.load_pipe()

        # run model for different configs
        for config in self.configs:
            if cached[config]:
                logging.info(f"Results already exist for {config} config, skipping")
                continue
            df = self.examples[config]
            # make config available for whole runner instance
            self.config = config
            # run model on examples
            logging.info(f"Running model {self.model_name} for {config} config")
            batch_size = self.batch_sizes()[self.model_name]
            if self.save_memory:
                batch_size = 1
            result_df = self.run_pipe(df, batch_size=batch_size, pipe=pipe, config=config)
            PATH = self.get_path(config)
            result_df.to_json(PATH)
    
    def convert_to_result(self, prediction_list):
        """converts a list of lists to a single string with duplicate entries removed"""
        predictions = {}
        # if # predictions was 1, only a hash is returned, convert it to a list
        if isinstance(prediction_list, dict):
            prediction_list = [prediction_list]
        for k in range(self.k_runs):
            predictions[f"prediction_{k}"] = prediction_list[k]['answer']
        return predictions

    def run_pipe(self, dataset, pipe, config, batch_size=2):
        preds = {}
        for i in range(self.k_runs):
            preds[f"prediction_{i}"] = []
        preds['input_id'] = []
        preds['input_length'] = []
        result_dataset = Dataset.from_dict(preds)
        for example, out in zip(dataset, tqdm(pipe(KeyDataset(dataset, f"qa_{config}"), batch_size=batch_size))):
            # get a prediction for every chunk in the batch
            # split predictions to columns
            item = self.convert_to_result(out)
            # # add the predictions to the dataset
            item['page_id'] = example['id']
            item['input_length'] = self.input_length
            result_dataset = result_dataset.add_item(item)

        return result_dataset
