
from ..abstract_runner import AbstractRunner


class CerebrasRunner(AbstractRunner):


    @staticmethod
    def start_prompt():
        return "The following text talks about a person but the person is referred to as <mask>."

    @staticmethod
    def end_prompt():
        return "The name of the person referred to as mask is NOT <mask>, it's full name is "

    @staticmethod
    def names():
        return {
            "cerebras-0b111": "cerebras/Cerebras-GPT-111M",
            "cerebras-1b3": "cerebras/Cerebras-GPT-1.3B",
            "cerebras-2b7": "cerebras/Cerebras-GPT-2.7B",
            "cerebras-6b7": "cerebras/Cerebras-GPT-6.7B",
            "cerebras-13b": "cerebras/Cerebras-GPT-13B",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "cerebras-0b111",
            "S": "cerebras-1b3",
            "M": "cerebras-2b7",
            "L": "cerebras-6b7",
            "XL": "cerebras-13b",
        }

    @staticmethod
    def batch_sizes():
        return {
            "cerebras-0b111": 64,
            "cerebras-1b3": 8,
            "cerebras-2b7": 8,
            "cerebras-6b7": 4,
            "cerebras-13b": 4,
        }