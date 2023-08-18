"""masking of the rulings dataset"""

from datasets import Dataset, load_dataset
import re

# TODO: add language for xlm models to set default language
# TODO: some type of paraphrasing to make text shorter

class RulingsPreparer():

    def __init__(self, dataset: Dataset) -> Dataset:
        self.mask_regex = r"A._"

    @staticmethod
    def select_test_set_ids():
        # finds well suited court rulings for testing and saves their id to a file
        dataset = load_dataset("rcds/swiss_rulings", split="train")

        # filter for rulings from the swiss federal court
        dataset = dataset.filter(lambda x: x["court"] == "CH_BGer", num_proc=8)

        # filter for rulings from 2019
        dataset = dataset.filter(lambda x: x["year"] == 2019, num_proc=8)

        # extract ids and save to file
        ids = dataset["decision_id"]
        with open("test_set_ids_rulings.csv", "w") as f:
            for id in ids:
                f.write(f"{id}\n")
    
    def prepare_rulings(self, dataset):
        # use id as identifier for consistency
        dataset = dataset.rename_column("decision_id", "id")
        # mask texts
        dataset = self.mask_dataset(dataset)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def mask_example(self, example):
        """replaces all occurences of the mask_regex with the mask token <mask>"""
        example["masked_text_original"] = re.sub(self.mask_regex, "<mask>", example["full_text"])
        return example
    
    def remove_unnecessary_columns(self, dataset):
        required_columns = ["id", "masked_text_original", "language"]
        remove = [column for column in dataset.column_names if column not in required_columns]
        dataset = dataset.remove_columns(remove)
        return dataset
    
    def mask_dataset(self, dataset):
        dataset = dataset.map(self.mask_example)
        return dataset

def main():
    RulingsPreparer.select_test_set_ids()

if __name__ == "__main__":
    main()