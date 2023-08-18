from .abstract_runner import AbstractRunner
import logging
import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class AbstractTextToTextRunner(AbstractRunner):

    def get_model(self):
        """retrieves model from huggingface model hub and load it to specified device"""
        logging.info(f"Loading model for {self.model_name}")
        model_path = self.names()[self.model_name]
        # if GPU is available, load in 8bit mode
        if torch.cuda.is_available():
            return self._model_loader().from_pretrained(model_path, load_in_8bit=True, device_map="auto")
        else:
            logging.warning("GPU not available, loading model in FP32 mode on CPU. This will be very slow.")
            return self._model_loader().from_pretrained(model_path)
    
    def get_tokenizer(self):
        logging.info(f"Loading tokenizer for {self.model_name}")
        model_path = self.names()[self.model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True, padding="longest")
        return tokenizer
    
    @staticmethod
    def _model_loader():
        return AutoModelForSeq2SeqLM

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
            result_df = df.map(self.make_predictions, batched=True, batch_size=batch_size, remove_columns=df.column_names,
                               fn_kwargs={'k_runs': self.k_runs, 'config': self.config})        
            PATH = self.get_path(config)
            result_df.to_json(PATH)

    def make_predictions(self, examples, config, k_runs=1):
        # tokenize inputs and move to GPU
        texts = examples[f"masked_text_{config}"]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        # compute lengths of the inputs to store with the result
        input_lengths = [len(i) for i in examples[f"masked_text_{config}"]]
        pad_token = self.tokenizer.eos_token_id

        predictions = {}
        # generate predictions
        generated_ids = self.model.generate(**inputs, early_stopping=True, num_beams=5,
                                            num_return_sequences=k_runs, pad_token_id=pad_token, max_new_tokens=5)
        # decode predictions
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # split outputs into len(inputs) lists to store them as independent predictions
        result = [outputs[i * k_runs: (i + 1) * k_runs] for i in range(len(texts))]

        for i in range(k_runs):
            predictions[f"prediction_{i}"] = []
        for k, generated_responses in enumerate(result):
            # for every generated sequence for this example
            for i, out in enumerate(generated_responses):
                predictions[f"prediction_{i}"].append(out)

        # add page_id and input_length to the result
        predictions["page_id"] = examples["id"]
        predictions["input_length"] = input_lengths
        return predictions