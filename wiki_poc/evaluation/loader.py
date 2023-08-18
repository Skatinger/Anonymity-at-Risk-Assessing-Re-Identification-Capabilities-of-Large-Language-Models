
# run from wiki_poc with `python -m evalution.loader`

import os
import re
import logging
from models.model_runner import get_all_model_names, load_test_set
from datasets import load_dataset
import json


class ResultLoader():

    def result_file_names(self, key):
        return os.listdir(f"results/{key}")

    def load_gt(self):
        return load_test_set()
    
    def load_rulings_gt(self, dataset_type = "rulings"):
        return load_test_set(dataset_type=dataset_type)

    def load(self, key, model_name = None, model_class = None):
        """loads and returns all results as dict. can be filtered by model_class or model_name
           returns: {"model-name": {"..., data: Dataset"}} 
        """
        file_names = self.result_file_names(key)
        if model_name:
            return self.load_by_name(key, model_name)
        if model_class:
            return self.load_by_class(key, model_class)

        models = {}
        for file_name in file_names:
            name, size, config, inputsize = self.parse_name(file_name)
            logging.info(f"loading dataset from: results/{key}/{file_name}")

            if not models.get(name):
                models[name] = {}
            
            if not models.get(name).get(size):
                models[name][size] = {
                    "size": float(re.sub('b', '.', size)), # store size as float
                    "inputsize": inputsize
                }
            
            models[name][size][config] = load_dataset("json", data_files=f"results/{key}/{file_name}")

        if len(models) == 0:
            logging.warning(f"No results found for key {key}")
        return models
    
    def load_by_class(self, model_class):
        pass

    def load_by_name(self, key, model_name):
        file_names = self.result_file_names(key)
        models = {}
        for file_name in file_names:
            # skip file if it doesn't match the model name
            if not file_name.startswith(model_name):
                continue
            name, size, config, inputsize = self.parse_name(file_name)
            logging.info(f"loading dataset from: results/{key}/{file_name}")

            if not models.get(name):
                models[name] = {}
            
            if not models.get(name).get(size):
                models[name][size] = {
                    "size": float(re.sub('b', '.', size)), # store size as float
                    "inputsize": inputsize
                }
            
            models[name][size][config] = load_dataset("json", data_files=f"results/{key}/{file_name}")
        if len(models) == 0:
            logging.warning(f"No results found for model {model_name} and key {key}")
        return models

    def parse_name(self, filename):
        model_name, rest = filename.split("-")  # Split the filename by underscores
        size, config, inputsize = rest.split("_")  # Split the remaining part by hyphen to get size, config, and inputsize
        inputsize = inputsize.rstrip(".json")  # Remove the ".json" extension from inputsize
        return model_name, size, config, inputsize

    def load_computed(self, key, model_name):
        """loads and returns all computed results as dict. can be filtered by model_name and config"""
        if model_name:
            filepath = f"evaluation/results/{key}-{model_name}-results.json"
        else:
            filepath = f"evaluation/results/{key}-results.json"
        print(filepath)
        if not os.path.exists(filepath):
            logging.warning(f"File {filepath} does not exist")
            return {}
        # load json file
        json_file = open(filepath)
        data = json.load(json_file)
        json_file.close()
        return data



def main():
    loader = ResultLoader()
    dict = loader.load("1000_basic")
    print(dict)
    

if __name__ == "__main__":
    main()
