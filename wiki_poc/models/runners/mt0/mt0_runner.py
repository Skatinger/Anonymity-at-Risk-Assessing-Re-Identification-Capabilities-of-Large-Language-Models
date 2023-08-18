
from ..abstract_text_to_text_runner import AbstractTextToTextRunner

class MT0Runner(AbstractTextToTextRunner):

    @staticmethod
    def names():
        return {
            "mt0-0b300": "bigscience/mt0-small",
            "mt0-0b580": "bigscience/mt0-base",
            "mt0-1b2": "bigscience/mt0-large",
            "mt0-3b7": "bigscience/mt0-xl",
            "mt0-13b": "bigscience/mt0-xxl",
        }
    
    @staticmethod
    def sizes():
        return {
            "XS": "mt0-0b300",
            "S": "mt0-0b580",
            "M": "mt0-1b2",
            "L": "mt0-3b7",
            "XL": "mt0-13b",
        }
        
    @staticmethod
    def batch_sizes():
        return {
            "mt0-0b300": 32,
            "mt0-0b580": 16,
            "mt0-1b2": 8,
            "mt0-3b7": 4,
            "mt0-13b": 2,
        }