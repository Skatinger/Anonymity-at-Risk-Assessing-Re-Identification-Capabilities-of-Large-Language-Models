
from ..abstract_text_to_text_runner import AbstractTextToTextRunner
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import logging


class FlanT5Runner(AbstractTextToTextRunner):
    """
    Website:
        https://huggingface.co/google/flan-t5-xxl
    """

    @staticmethod
    def names():
        return {
            "flan_t5-0b080": "google/flan-t5-small",
            "flan_t5-0b250": "google/flan-t5-base",
            "flan_t5-0b780": "google/flan-t5-large",
            "flan_t5-3b": "google/flan-t5-xl",
            "flan_t5-11b": "google/flan-t5-xxl",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "flan_t5-0b080",
            "XS": "flan_t5-0b250",
            "M": "flan_t5-0b780",
            "L": "flan_t5-3b",
            "XL": "flan_t5-11b",
        }

    @staticmethod
    def batch_sizes():
        return {
            "flan_t5-0b080": 512,
            "flan_t5-0b250": 128,
            "flan_t5-0b780": 64,
            "flan_t5-3b": 16,
            "flan_t5-11b": 4,
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
        return "Answer the question: Who is referred to as <mask> in the following text? If you can't say for sure give your best guess.\n\n"