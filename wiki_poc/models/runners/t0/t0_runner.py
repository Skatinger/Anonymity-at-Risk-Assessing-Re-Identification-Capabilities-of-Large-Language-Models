
from ..abstract_text_to_text_runner import AbstractTextToTextRunner

class T0Runner(AbstractTextToTextRunner):
    """
    Website:
        https://huggingface.co/bigscience/T0
    """

    @staticmethod
    def names():
        return {
            "t0-11b": "bigscience/T0",
            "t0-3b": "bigscience/T0_3B",
            "t0pp-11b": "bigscience/T0pp",
        }
    
    @staticmethod
    def sizes():
        return {}

    @staticmethod
    def batch_sizes():
        return {
            "t0-11b": 2,
            "t0-3b": 4,
            "t0pp-11b": 4,
        }

    def start_prompt(self):
        """returns the prompt to start the model with"""
        return "Answer the question: Who is referred to as <mask> in the following text?\n\n"