
from ..abstract_fill_mask_runner import AbstractFillMaskRunner
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import logging

class BertRunner(AbstractFillMaskRunner):

    @staticmethod
    def names():
        return {
            "swiss_bert-0b110": "ZurichNLP/swissbert",
            "xlm_swiss_bert-0b110": "ZurichNLP/swissbert-xlm-vocab",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "swiss_bert-0b110",
        }
    @staticmethod
    def batch_sizes():
        return {
            "swiss_bert-0b110": 64,
            "xlm_swiss_bert-0b110": 64,
        }
    
    def get_tokenizer(self):
        tokenizer = super().get_tokenizer()
        if self.model_name == "xlm_swiss_bert-0b110":
            # this model does not have a maximum input length as default, so we set it manually...
            tokenizer.model_max_length = 512
        return tokenizer
    
    def run_pipe(self, dataset, pipe, config, batch_size=2):
        """overwrite run_pipe to allow setting the correct language header"""
        if not "language" in dataset.column_names:
            logging.warning("No language column found in dataset, using default language de_CH.")
            pipe.model.set_default_language("de_CH")
            return super().run_pipe(dataset, pipe, config, batch_size)

        preds = {}
        for i in range(self.k_runs):
            preds[f"prediction_{i}"] = []
        preds['input_id'] = []
        preds['input_length'] = []
        preds['language'] = []
        result_dataset = Dataset.from_dict(preds)
        languages = {
            "de": "de_CH",
            "fr": "fr_CH",
            "it": "it_CH",
        }
        language_filtered_dataset = {}

        # convert dataset to dataset shard for each language
        for language, language_name in languages.items():
            language_filtered_dataset[language] = dataset.filter(lambda x: x['language'] == language)

        for language, filtered_dataset in language_filtered_dataset.items():
            pipe.model.set_default_language(languages[language])
            for example, out in zip(dataset, tqdm(pipe(KeyDataset(filtered_dataset, f"masked_text_{config}"), batch_size=batch_size))):
                # get a prediction for every chunk in the batch
                tokens, _scores = self.extract_result(out)
                # split predictions to columns
                item = self.convert_to_result(tokens)
                # # add the predictions to the dataset
                item['page_id'] = example['id']
                item['language'] = example['language']
                item['input_length'] = self.input_length
                result_dataset = result_dataset.add_item(item)

        return result_dataset