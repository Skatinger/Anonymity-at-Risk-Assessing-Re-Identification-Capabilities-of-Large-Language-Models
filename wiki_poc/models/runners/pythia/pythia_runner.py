
from ..abstract_runner import AbstractRunner
from transformers import GPTNeoXForCausalLM

class PythiaRunner(AbstractRunner):

    @staticmethod
    def __model_loader():
        return GPTNeoXForCausalLM

    @staticmethod
    def start_prompt():
        return "The following text talks about a person but the person is referred to as <mask>."

    @staticmethod
    def end_prompt():
        return "The name of the person referred to as mask is NOT <mask>. The correct name is: "

    @staticmethod
    def names():
        return {
            "pythia-0b070": "EleutherAI/pythia-70m",
            "pythia-0b160": "EleutherAI/pythia-160m",
            "pythia-0b410": "EleutherAI/pythia-410m",
            # "pythia-1b": "EleuterAI/pythia-1b", # model not available, is password protected
            "pythia-1b4": "EleutherAI/pythia-1.4b",
            "pythia-2b8": "EleutherAI/pythia-2.8b",
            "pythia-6b9": "EleutherAI/pythia-6.9b",
            "pythia-12b": "EleutherAI/pythia-12b",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "pythia-0b410",
            # "XS": "pythia-1b",
            "S": "pythia-1b4",
            "M": "pythia-2b8",
            "L": "pythia-6b9",
            "XL": "pythia-12b",
        }

    @staticmethod
    def batch_sizes():
        return {
            "pythia-0b070": 128,
            "pythia-0b160": 64,
            "pythia-0b410": 32,
            # "pythia-1b": 8,
            "pythia-1b4": 8,
            "pythia-2b8": 8,
            "pythia-6b9": 4,
            "pythia-12b": 4,
        }