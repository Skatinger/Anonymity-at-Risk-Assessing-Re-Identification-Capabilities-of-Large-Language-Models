

from ..abstract_fill_mask_runner import AbstractFillMaskRunner
from transformers import AutoModelForMaskedLM, AutoTokenizer
import logging

class XLMRobertaRunner(AbstractFillMaskRunner):

    @staticmethod
    def names():
        return {
            "legal_swiss_longformer-0b279": "joelito/legal-swiss-longformer-base", # rulings
            "legal_xlm_longformer-0b279": "joelito/legal-xlm-longformer-base", # rulings
            "legal_swiss_roberta-0b279": "joelito/legal-swiss-roberta-base", # rulings
            "legal_swiss_roberta-0b561": "joelito/legal-swiss-roberta-large", # rulings
            "legal_xlm_roberta-0b279": "joelito/legal-xlm-roberta-base", # rulings
            "legal_xlm_roberta-0b561": "joelito/legal-xlm-roberta-large", # rulings
        }
    
    @staticmethod
    def sizes():
        logging.info("No sizes for XLM-Roberta runner defined")
        return {}

    @staticmethod
    def batch_sizes():
        return {
            "legal_swiss_longformer-0b279": 16,
            "legal_xlm_longformer-0b279": 16,
            "legal_swiss_roberta-0b279": 16,
            "legal_swiss_roberta-0b561": 16,
            "legal_xlm_roberta-0b279": 16,
            "legal_xlm_roberta-0b561": 16,
        }

    def _model_loader(self):
        return AutoModelForMaskedLM

    def _tokenizer_loader(self):
        return AutoTokenizer
