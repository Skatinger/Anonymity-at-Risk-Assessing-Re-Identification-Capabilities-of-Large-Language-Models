import logging
import transformers
import torch
from transformers import AutoTokenizer

from ..abstract_runner import AbstractRunner

class MPTRunner(AbstractRunner):

    @staticmethod
    def names():
        return {
            "mpt-7b": "mosaicml/mpt-7b"
        }
    
    @staticmethod
    def sizes():
        return {
            "L": "mpt-7b",
        }

    @staticmethod
    def batch_sizes():
        return {
            "mpt-7b": 16,
        }

    def get_tokenizer(self):
        logging.info(f"Loading tokenizer for {self.model_name}")
        # model was trained with gpt-neox-20b, so we use that tokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", padding_side="left")
        # have to add a padding token, as processing is done in batches and required inputs to be padded,
        # but this tokenizer does not have a pad token by default.
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer

    def get_model(self):
        """retrieves model from huggingface model hub and load it to specified device"""
        logging.info(f"Loading model for {self.model_name}")
        model_path = self.names()[self.model_name]

        # requires `trust_remote_code` as not all parts are yet included in the transformers library
        config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.attn_config['attn_impl'] = 'triton'

        if torch.cuda.is_available():
            return transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to(self.device)
        else:
            logging.warning("GPU not available, loading model in FP32 mode on CPU. This will be very slow.")
            return transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
