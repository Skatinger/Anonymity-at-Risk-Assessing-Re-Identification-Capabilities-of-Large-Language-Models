import logging
from ..abstract_runner import AbstractRunner


class MajorityNameRunner(AbstractRunner):
    """predicts the most common names in the english language
       can select firstname, lastname and fullname
    """

    def set_options(self, options):
        if "input_length" in options:
            self.input_length = options["input_length"]
        if "k_runs" in options:
            self.k_runs = options["k_runs"]
        if "save_memory" in options:
            self.save_memory = options["save_memory"]
        if "device" in options:
            self.device_number = options["device"]
        if "configs" in options:
            self.configs = options["configs"]
        self.options = options
    
    @staticmethod
    def names():
        return {
            "majority_full_name-1b": "random_full_name",
            # "random_first_name-1b": "random_first_name",
            # "random_last_name": "random_last_name",
        }
    
    @staticmethod
    def sizes():
        return {}
    
    @staticmethod
    def batch_sizes():
        return {
            "majority_full_name-1b": 1024,
            # "random_first_name": 1024,
            # "random_last_name": 1024,
        }


    def first_names(self):
        # https://www.ssa.gov/oact/babynames/decades/century.html
        return [
            "James",
            "Robert",
            "John",
            "Michael",
            "David",
        ]
    
    def last_names(self):
        # https://en.wiktionary.org/wiki/Appendix:English_surnames_(England_and_Wales)
        return [
            "Smith",
            "Jones",
            "Williams",
            "Taylor",
            "Brown",
        ]
    
    def full_names(self):
        return [f"{firstname} {lastname}" for firstname, lastname in zip(self.first_names(), self.last_names())]
    
    def start_prompt(self):
        return ""

    def end_prompt(self):
        return ""

    def run_model(self):
        # check if results already exist
        cached = self.check_cache()
        if all(cached.values()):
            logging.info(f"Results already exist, skipping model {self.model_name}")
            return
        # prepare examples for different configs
        self.prepare_examples()

        # run model for different configs
        for config in self.configs:
            if cached[config]:
                logging.info(f"Results already exist for {config} config, skipping")
                continue
            df = self.examples[config]
            # make config available for whole runner instance
            self.config = config
            # run model on examples
            logging.info(f"Running baseline {self.model_name} for {config} config")
            batch_size = self.batch_sizes()[self.model_name]
            result_df = df.map(self.make_predictions, batched=True, batch_size=batch_size, remove_columns=df.column_names,
                               fn_kwargs={'k_runs': self.k_runs, 'config': self.config})
            PATH = self.get_path(config)
            result_df.to_json(PATH)
    
    def make_predictions(self, examples, config, k_runs=1):
        predictions = {}
        input_lengths = [len(i) for i in examples[f"masked_text_{config}"]]
        # fill names for all predictions
        for i in range(k_runs):
            name = self.full_names()[i]
            predictions[f"prediction_{i}"] = [name] * len(examples["id"])
        # fill page_id and input_length
        predictions["page_id"] = examples["id"]
        predictions["input_length"] = input_lengths
        return predictions
        


