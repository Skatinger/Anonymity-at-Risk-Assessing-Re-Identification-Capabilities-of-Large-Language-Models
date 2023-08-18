
from ..abstract_runner import AbstractRunner

class GPTNeoXRunner(AbstractRunner):


    @staticmethod
    def names():
        return {
            "gpt_neox-20b": "EleutherAI/gpt-j-6b",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXL": "gpt_neox-20b",
        }
        
    @staticmethod
    def batch_sizes():
        return {
            "gpt_neox-20b": 2,
        }

    @staticmethod
    def load_model(model_name):
        from transformers import GPTNeoForCausalLM
        # try loading in half precision, as it was trained this way.
        # if it doesn't fit, use 4bit quantization from bitsandbytes
        return GPTNeoForCausalLM.from_pretrained(model_name).half().cuda()

    @staticmethod
    def load_tokenizer(model_name):
        from transformers import GPTNeoTokenizerFast
        return GPTNeoTokenizerFast.from_pretrained(model_name)