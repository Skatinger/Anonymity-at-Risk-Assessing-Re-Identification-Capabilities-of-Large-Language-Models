
from ..abstract_qa_runner import AbstractQARunner

class RobertaQARunner(AbstractQARunner):

    @staticmethod
    def names():
        return {
            "roberta_squad-0b125": "deepset/roberta-base-squad2",
            "roberta_squad-0b355": "deepset/roberta-large-squad2",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "roberta_squad-0b125",
            "XS": "roberta_squad-0b355",
        }

    @staticmethod
    def batch_sizes():
        return {
            "roberta_squad-0b125": 64,
            "roberta_squad-0b355": 64,
        }