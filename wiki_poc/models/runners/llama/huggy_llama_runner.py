from ..abstract_runner import AbstractRunner

class HuggyLlamaRunner(AbstractRunner):

    @staticmethod
    def names():
        return {
            "llama-7b": "huggyllama/llama-7b",
            "llama-13b": "huggyllama/llama-13b",
            "llama-30b": "huggyllama/llama-30b",
            "llama-65b": "huggyllama/llama-65b",
        }
    
    @staticmethod
    def sizes():
        return {
            "L": "llama-7b",
            "XL": "llama-13b",
            "XXL": "llama-30b",
            "XXXL": "llama-65b",
        }
        
    @staticmethod
    def batch_sizes():
        return {
            "llama-7b": 8,
            "llama-13b": 8,
            "llama-30b": 4,
            "llama-65b": 4,
        }