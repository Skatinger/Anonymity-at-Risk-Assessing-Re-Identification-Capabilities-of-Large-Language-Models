
from ..abstract_text_to_text_runner import AbstractTextToTextRunner
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch
import logging


class MT5Runner(AbstractTextToTextRunner):

    @staticmethod
    def names():
        return {
            "mt5-0b300": "google/mt5-small",
            "mt5-0b580": "google/mt5-base",
            "mt5-1b2": "google/mt5-large",
            "mt5-3b7": "google/mt5-xl",
            "mt5-13b": "google/mt5-xxl",
        }
    
    @staticmethod
    def sizes():
        return {
            "XS": "mt5-0b300",
            "S": "mt5-0b580",
            "M": "mt5-1b2",
            "L": "mt5-3b7",
            "XXL": "mt5-13b",
        }

    @staticmethod
    def batch_sizes():
        return {
            "mt5-0b300": 64,
            "mt5-0b580": 32,
            "mt5-1b2": 16,
            "mt5-3b7": 8,
            "mt5-13b": 2,
        }

    def get_model(self):
        """retrieves model from huggingface model hub and load it to specified device"""
        logging.info(f"Loading model for {self.model_name}")
        model_path = self.names()[self.model_name]
        # if GPU is available, load in 8bit mode
        if torch.cuda.is_available():
            return MT5ForConditionalGeneration.from_pretrained(model_path, load_in_8bit=True, device_map="auto")
        else:
            logging.warning("GPU not available, loading model in FP32 mode on CPU. This will be very slow.")
            return MT5ForConditionalGeneration.from_pretrained(model_path)
    
    def get_tokenizer(self):
        logging.info(f"Loading tokenizer for {self.model_name}")
        model_path = self.names()[self.model_name]
        tokenizer = MT5Tokenizer.from_pretrained(model_path, truncation=True, padding="longest", max_length="model_max_length")
        return tokenizer

    def start_prompt(self):
        """returns the prompt to start the model with"""
        return "Answer the question: Who is referred to as <mask> in the following text?\n\n"