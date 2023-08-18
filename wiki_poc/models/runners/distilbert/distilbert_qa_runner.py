from ..abstract_qa_runner import AbstractQARunner
import torch
import logging
from transformers import DistilBertTokenizer, pipeline

class DistilbertQARunner(AbstractQARunner):


    @staticmethod
    def names():
        return {
            "distilbert_squad-0b062": "distilbert-base-cased-distilled-squad",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "distilbert_squad-0b062",
        }

    @staticmethod
    def batch_sizes():
        return {
            "distilbert_squad-0b062": 128,
        }

    def get_model(self):
        """override as device_map='auto' is not supported by distilbert"""
        if torch.cuda.is_available():
            logging.info(f"Loading model for {self.model_name}")
            model_path = self.names()[self.model_name]
            return self._model_loader().from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)
        else:
            super().get_model()

    def get_tokenizer(self):
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
        return tokenizer

    def load_pipe(self):
        logging.info(f"Loading pipeline for {self.model_name}")
        if not torch.cuda.is_available():
            logging.warning("GPU not available, loading pipeline in FP32 mode on CPU. This will be very slow.")
        # pipeline is automatically loaded on GPU if available when loading the model in 8bit mode
        return pipeline('question-answering', model=self.names()[self.model_name], top_k=self.k_runs)
