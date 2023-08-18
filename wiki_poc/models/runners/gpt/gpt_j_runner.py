
from ..abstract_runner import AbstractRunner


class GPTJRunner(AbstractRunner):


    @staticmethod
    def names():
        return {
            "gptj-6b": "EleutherAI/gpt-j-6b",
        }
    
    @staticmethod
    def sizes():
        return {
            "XL": "gptj-6b",
        }
        
    @staticmethod
    def batch_sizes():
        return {
            "gptj-6b": 8,
        }

    @staticmethod
    def end_prompt():
        return "\n\nWrite the name of the person referred to as <mask>. <mask> is not a valid name."