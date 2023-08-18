from .abstract_runner import AbstractRunner
import logging
import torch
from typing import Dict, List, Tuple, Union
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM
from transformers.pipelines.pt_utils import KeyDataset


class AbstractFillMaskRunner(AbstractRunner):

    def get_model(self):
        """retrieves model from huggingface model hub and load it to specified device
           overwrites AbstractRunner.get_model() as fillmask models are easier to run
           with the pipeline API instead of the model API, but the pipeline does not
           support 8bit mode. Therefore load the model in FP32."""
        logging.info(f"Loading model for {self.model_name}")
        model_path = self.names()[self.model_name]
        # if GPU is available pipeline will run on GPU
        if torch.cuda.is_available():
            return self._model_loader().from_pretrained(model_path)
        else:
            logging.warning("GPU not available, loading model in FP32 mode on CPU. This will be very slow.")
            return self._model_loader().from_pretrained(model_path)
    
    def _model_loader(self):
        return AutoModelForMaskedLM

    def get_tokenizer(self):
        logging.info(f"Loading tokenizer for {self.model_name}")
        model_path = self.names()[self.model_name]
        tokenizer = self._tokenizer_loader().from_pretrained(model_path, truncation=True, padding="longest", padding_side="left")
        return tokenizer
    
    def prepare_examples(self):
        """shortens input text to max length given and replaces masks in the text with the correct
           mask for the model used. (e.g. <mask>, <MASK>, etc.)
           # TODO: don't prepare examples for cached results
        """
        logging.info(f"Preparing examples for {self.model_name}")
        self.examples = {}
        for config in self.configs:
            # shorten input text to max length given
            df = self.dataset.map(lambda x: {f"masked_text_{config}": x[f"masked_text_{config}"][:self.input_length]}, num_proc=8)
            # remove all examples which do no longer contain a mask
            df = df.filter(lambda x: '<mask>' in x[f"masked_text_{config}"], num_proc=8)
            # for the fill mask models, the FillmaskPipelineWithTruncation is used, which truncates the input text,
            # which can lead to the mask token being truncated. Tokenize the input text here to ensure the mask is
            # not truncated.
            logging.info(f"Checking if mask token is truncated for {config}, this may take a while...")
            df = df.filter(lambda x: self.has_mask_after_truncation(x[f"masked_text_{config}"]), num_proc=8)
            # convert mask tokens to mask token format used by the model
            if self.tokenizer.mask_token != '<mask>':
                df = df.map(lambda x: {f"masked_text_{config}": x[f"masked_text_{config}"].replace('<mask>', self.tokenizer.mask_token)}, num_proc=8)
            self.examples[config] = df

    def has_mask_after_truncation(self, text):
        """returns true or false if mask token is in input_ids after truncation"""
        tokenized = self.tokenizer(text, return_tensors="pt", truncation=True, padding='max_length')
        # check if mask token is in input_ids
        mask_token_id = self.tokenizer.mask_token_id
        return mask_token_id in tokenized['input_ids']

    def load_pipe(self):
        logging.info(f"Loading pipeline for {self.model_name}")
        # specify gpu if available
        if torch.cuda.is_available():
            return FillMaskPipelineWithTruncation(model=self.model, tokenizer=self.tokenizer, top_k=self.k_runs, device=self.device)
        else:
            logging.warning("GPU not available, loading pipeline in FP32 mode on CPU. This will be very slow.")
            return FillMaskPipelineWithTruncation(model=self.model, tokenizer=self.tokenizer, top_k=self.k_runs, device=-1)

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
    
    def convert_to_result(self, lists):
        # """converts a list of lists to a single string with duplicate entries removed"""
        predictions = {}
        for i in range(self.k_runs):
            predictions[f"prediction_{i}"] = []
        # collect every k-th prediction of all predicted masks, and concat them to a single string
        for k in range(self.k_runs):
            all_k_predictions = []
            # find the k-th prediction for all predictions
            for sublist in lists:
                # ensure model actually returned k predictions for a mask, if not, skip
                if len(sublist) <= k:
                    continue
                all_k_predictions.append(sublist[k].strip())
            # remove duplicates
            unique_predictions = list(set(all_k_predictions))
            # and join to string
            predictions[f"prediction_{k}"] = ' '.join(unique_predictions)
        return predictions

    def run_pipe(self, dataset, pipe, config, batch_size=2):
        preds = {}
        for i in range(self.k_runs):
            preds[f"prediction_{i}"] = []
        preds['input_id'] = []
        preds['input_length'] = []
        result_dataset = Dataset.from_dict(preds)
        for example, out in zip(dataset, tqdm(pipe(KeyDataset(dataset, f"masked_text_{config}"), batch_size=batch_size))):
            # get a prediction for every chunk in the batch
            tokens, _scores = self.extract_result(out)
            # split predictions to columns
            item = self.convert_to_result(tokens)
            # # add the predictions to the dataset
            item['page_id'] = example['id']
            item['input_length'] = self.input_length
            result_dataset = result_dataset.add_item(item)

        return result_dataset

    def extract_result(self, result: Union[Dict, List[Dict]]) -> Tuple[List[str], List[float]]:
        """extracts the relevant parts of the result of the fill-mask pipeline, removes unnecessary
        text outputs. Kept outputs are the predicted strings and their score. Score is rounded to 3 digits after comma

        Args:
            result: model output of the fill-mask pipeline

        Returns:
            Tuple: of two lists, one containing the predicted strings, the other containing the scores
        """
        # if we get no predictions, return empty arrays
        if len(result) < 1:
            return [], []
        # result is an array with results for each input sequence. If there was only a single input sequence, the array
        # will only contain a single element.
        # each input sequence has a result for each mask, e.g a sequence with 2 masks will have 2 results in the array
        # representing the sequence results
        # for each result of a mask, there are five predictions with token and score
        results_tokens = []
        results_scores = []
        # iterate over all processed sequences
        for sequence_results in result:
            # if there was only a single mask, the result is not an array of arrays with each array containing 5
            # prediction hashes, but a single array containing 5 prediction hashes. This is why we need to check if
            # the result is an array of arrays and convert it to an array of arrays if it is not

            # first check if the sequence result itself is an array. If it is not, we need to convert it to an array
            if not isinstance(sequence_results, list):
                sequence_results = [sequence_results]

            # then, after the sequence result is definitely an array, check if the first element is an array.
            # if it is not, this was a sequence with a single mask. We need to convert the array to an array of arrays
            if not isinstance(sequence_results[0], list):
                sequence_results = [sequence_results]

            # iterate over all masks in the sequence, e.g. the predictions for those masks
            for mask_result in sequence_results:
                # if only a single mask was present in the sequence, it's just a dict, without array.
                # Wrap it in this case to make the following code work for both cases
                if isinstance(mask_result, dict):
                    mask_result = [mask_result]

                # format of mask_result should be: [{token_str: predicted token, score: predicted sore, ...}, {...}]
                # we only need the token_str and score, so we extract those and append them to the results
                results_tokens.append([prediction['token_str'] for prediction in mask_result])
                results_scores.append([round(prediction['score'], 3) for prediction in mask_result])
        # return format is a an array of length of input sequences, with each array
        # containing an array of 5 predictions for each mask, e.g.
        # [sequence1_results, ...] where sequence1_results = [results_for_mask1, ...] where
        # results_for_mask1 = [(token_str, score), (...), ...]
        return results_tokens, results_scores



# small helper class
from transformers.pipelines import FillMaskPipeline
from transformers.pipelines.base import GenericTensor
class FillMaskPipelineWithTruncation(FillMaskPipeline):
    """Overwrite the fill-mask pipeline to allow for truncation of the input text and parsing preprocessing parameters.
    Args:
        same arguments as FillMaskPipeline, extended with truncation and padding parameters for the tokenizer.
    Returns:
        same as FillMaskPipeline
    """    
    def preprocess(self, inputs, return_tensors=None, **preprocess_parameters) -> Dict[str, GenericTensor]:
        """Overwrites the preprocess method of the fill-mask pipeline to allow for truncation of the input text."""
        preprocess_parameters["truncation"] = True
        preprocess_parameters["padding"] = 'max_length'
        if return_tensors is None:
            return_tensors = self.framework
        model_inputs = self.tokenizer(inputs, return_tensors=return_tensors, truncation=True, padding='max_length')
        # ensure there is still a mask after truncation
        self.ensure_exactly_one_mask_token(model_inputs)
        return model_inputs