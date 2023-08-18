

from ..abstract_fill_mask_runner import AbstractFillMaskRunner
from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer
import torch
import logging

class DebertaRunner(AbstractFillMaskRunner):

    @staticmethod
    def names():
        return {
            # "mdeberta-0b086": "microsoft/mdeberta-v3-base",
            "deberta-0b304": "microsoft/deberta-v3-large",
            # "deberta-sadf": "microsoft/deberta-large",
            # "deberta": "microsoft/deberta-v2-xlarge"
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "mdeberta-0b086",
            "XS": "deberta-0b304",
        }

    @staticmethod
    def batch_sizes():
        return {
            "mdeberta-0b086" : 32,
            "deberta-0b304": 32,
        }

    def _model_loader(self):
        return DebertaV2ForMaskedLM

    def get_model(self):
        """override as device_map='auto' is not supported by distilbert"""
        if torch.cuda.is_available():
            logging.info(f"Loading model for {self.model_name}")
            model_path = self.names()[self.model_name]
            return self._model_loader().from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)
        else:
            super().get_model()
