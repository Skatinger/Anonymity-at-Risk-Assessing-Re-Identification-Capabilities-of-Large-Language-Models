
from ..abstract_text_to_text_runner import AbstractTextToTextRunner
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import logging


class T5Runner(AbstractTextToTextRunner):
    """
    Website:
        https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Model
    """

    @staticmethod
    def names():
        return {
            "t5-0b060": "t5-small",
            "t5-0b220": "t5-base",
            "t5-0b770": "t5-large",
            "t5-3b": "t5-3b",
            "t5-11b": "t5-11b",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "t5-0b060",
            "XS": "t5-0b220",
            "S": "t5-0b770",
            "L": "t5-3b",
            "XL": "t5-11b",
        }

    @staticmethod
    def batch_sizes():
        return {
            "t5-0b060": 64,
            "t5-0b220": 64,
            "t5-0b770": 32,
            "t5-3b": 16,
            "t5-11b": 4,
        }

    def get_model(self):
        """retrieves model from huggingface model hub and load it to specified device"""
        logging.info(f"Loading model for {self.model_name}")
        model_path = self.names()[self.model_name]
        # if GPU is available, load in 8bit mode
        if torch.cuda.is_available():
            return T5ForConditionalGeneration.from_pretrained(model_path, load_in_8bit=True, device_map="auto")
        else:
            logging.warning("GPU not available, loading model in FP32 mode on CPU. This will be very slow.")
            return T5ForConditionalGeneration.from_pretrained(model_path)
    
    def get_tokenizer(self):
        logging.info(f"Loading tokenizer for {self.model_name}")
        model_path = self.names()[self.model_name]
        tokenizer = T5Tokenizer.from_pretrained(model_path, truncation=True, padding="longest", max_length="model_max_length")
        return tokenizer

    def start_prompt(self):
        """returns the prompt to start the model with"""
        return "Answer the question: Who is referred to as <mask> in the following text?\n\n"