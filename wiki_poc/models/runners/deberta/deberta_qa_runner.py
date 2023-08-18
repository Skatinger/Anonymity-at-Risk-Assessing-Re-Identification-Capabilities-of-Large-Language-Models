

from ..abstract_qa_runner import AbstractQARunner
import torch
import logging
class DebertaQARunner(AbstractQARunner):

    @staticmethod
    def names():
        return {
            "deberta_squad-0b100": "deepset/deberta-v3-large-squad2",
        }
    
    @staticmethod
    def sizes():
        return {
            "XS": "deberta_squad-0b100",
        }

    @staticmethod
    def batch_sizes():
        return {
            "deberta_squad-0b100": 64,
        }

    def get_model(self):
        """override as device_map='auto' is not supported by deberta"""
        if torch.cuda.is_available():
            logging.info(f"Loading model for {self.model_name}")
            model_path = self.names()[self.model_name]
            return self._model_loader().from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)
        else:
            super().get_model()
