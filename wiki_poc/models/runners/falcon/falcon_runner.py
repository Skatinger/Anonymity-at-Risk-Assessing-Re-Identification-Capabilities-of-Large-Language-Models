
from ..abstract_runner import AbstractRunner
import logging
import torch

class FalconRunner(AbstractRunner):


    @staticmethod
    def start_prompt():
        return """
                Below is an instruction that describes a task. Write a response that appropriately completes the request.
                ### Instruction:
                The following text is an extract from a wikipedia page. The text is about a person but the person is referred to as <mask>.
                Please give the name of the person referred to as <mask> and only the name. If you don't know the name,
                give your best guess. Do not include any other information in your response, it should only be the name, nothing else.

                The text:

                """

    @staticmethod
    def end_prompt():
        return """

                ### Response:
                """

    @staticmethod
    def names():
        return {
            "falcon-40b": "tiiuae/falcon-40b",
            "falcon-7b": "tiiuae/falcon-7b",
        }
    
    @staticmethod
    def sizes():
        return {
            "XL": "falcon-7b",
            "XXL": "falcon-40b",
        }
        
    @staticmethod
    def batch_sizes():
        return {
            "falcon-7b": 8,
            "falcon-40b": 2,
        }

    def get_model(self):
        """retrieves model from huggingface model hub and load it to specified device"""
        logging.info(f"Loading model for {self.model_name}")
        model_path = self.names()[self.model_name]
        # if GPU is available, load in 8bit mode
        if torch.cuda.is_available():
            return self._model_loader().from_pretrained(
                model_path, load_in_8bit=True, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        else:
            logging.warning("GPU not available, cannot load this model.")
            exit(1)

    def make_predictions(self, examples, config, k_runs=1):
        # tokenize inputs and move to GPU
        texts = examples[f"masked_text_{config}"]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        # compute lengths of the inputs to store with the result
        input_lengths = [len(i) for i in examples[f"masked_text_{config}"]]
        # generate predictions
        pad_token = self.tokenizer.eos_token_id

        predictions = {}
        # model generates k sequences for each input, all concated to one list
        generated_ids = self.model.generate(
            **inputs, num_beams=k_runs, early_stopping=True, num_return_sequences=k_runs, pad_token_id=pad_token, max_new_tokens=5)
        # decode predictions
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # split outputs into len(inputs) lists to store them as independent predictions
        result = [outputs[i * k_runs: (i + 1) * k_runs] for i in range(len(texts))]
        # initialize predictions dict
        for i in range(k_runs):
            predictions[f"prediction_{i}"] = []
        # get prediction and remove the input from the output, append prediction to the result
        for k, generated_sequences in enumerate(result):
            # for every generated sequence for this example
            for i, out in enumerate(generated_sequences):
                predictions[f"prediction_{i}"].append(out.replace(examples[f"masked_text_{config}"][k], ""))
        
        # append additional info with page_id and input_length for each example
        predictions['page_id'] = examples['id']
        predictions['input_length'] = input_lengths
        return predictions
