
from ..abstract_runner import AbstractRunner


class BloomzRunner(AbstractRunner):

    @staticmethod
    def names():
        return {
            "bloomz-1b1": "bigscience/bloomz-1b1",
            "bloomz-1b7": "bigscience/bloomz-1b7",
            "bloomz-3b": "bigscience/bloomz-3b",
            "bloomz-7b1": "bigscience/bloomz-7b1",
        }
    
    @staticmethod
    def sizes():
        return {
            "XS": "bloomz-1b1",
            "S": "bloomz-1b7",
            "M": "bloomz-3b",
            "L": "bloomz-7b1",
        }
        
    @staticmethod
    def batch_sizes():
        return {
            "bloomz-1b1": 8,
            "bloomz-1b7": 8,
            "bloomz-3b": 4,
            "bloomz-7b1": 2,
        }