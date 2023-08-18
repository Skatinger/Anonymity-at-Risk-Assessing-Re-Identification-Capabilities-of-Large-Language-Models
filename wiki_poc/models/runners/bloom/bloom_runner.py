
from ..abstract_runner import AbstractRunner


class BloomRunner(AbstractRunner):

    @staticmethod
    def names():
        return {
            "bloom-1b1": "bigscience/bloom-1b1",
            "bloom-1b7": "bigscience/bloom-1b7",
            "bloom-3b": "bigscience/bloom-3b",
            "bloom-7b1": "bigscience/bloom-7b1",
        }
    
    @staticmethod
    def sizes():
        return {
            "XS": "bloom-1b1",
            "S": "bloom-1b7",
            "M": "bloom-3b",
            "L": "bloom-7b1",
        }
        
    @staticmethod
    def batch_sizes():
        return {
            "bloom-1b1": 16,
            "bloom-1b7": 16,
            "bloom-3b": 8,
            "bloom-7b1": 8,
        }