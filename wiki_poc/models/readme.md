# Running Models

To run any models use the provided model_runner.py. As with all modules in this repo running the module
will only work from the root of the project section, in this case `wiki_poc`.
The model_runner supports almost all thinkable options and handles tedious parts for you like loading and preparing
your datasets, loading and placing models on GPUs as well as caching and structured results tracking.

If you are unsure about something you can always come back to this documentation or use the --help option 
on the model_runner.


### Examples

- Running everything on the default dataset (wiki)
`python -m models.model_runner`

- Running with a folder for saving the results and a model class
`python -m models.model_runner --key my-preferred-folder-name --model-class bloom`

- Running two specific models, but only with a short run to check if things work
`python -m models.model_runner --model falcon-7b,mt0-13b --dry-run --key dry-run-folder-first-try`

- Running a custom dataset on a model selection with specific input length and beam_search strategy
`python -m models.model_runner --key legal-hand-picked --model legal_xlm_roberta-0b561,legal_swiss_roberta-0b561,swiss_bert-0b110,xlm_swiss_bert-0b110,legal_swiss_longformer-0b279,legal_xlm_longformer-0b279 --custom-dataset manually_reidentified_updated_with_paraphrased.csv --options input_length=10000,strategy=beam_search`

- Running with a specific number of sentences as input instead of a number of characters: (careful, if you use a custom
dataset make sure it supports this by checking that the same amount of sentences is present in each.)
`python -m models.model_runner --options input_sentences_count=5`

# available models
For a list of available models you can pass any invalid model name and receive the full list:
```
`>> python -m models.model_runner --model help`
('Model help does not exist. ', 'Please choose one of the following models: ', ['t0pp-11b', 'mt5-0b300', 't5-3b', 'cerebras-13b', 'bloomz-7b1', 't5-11b', 'bloomz-1b7', 'bloomz-1b1', 'legal_xlm_longformer-0b279', 'bloom-3b', 't0-3b', 'flan_t5-0b780', 'llama-13b', 'legal_swiss_roberta-0b279', 'llama-7b', 't5-0b060', 'legal_swiss_roberta-0b561', 'legal_swiss_longformer-0b279', 'falcon_instruct-7b', 'random_full_name-1b', 'pythia-0b070', 't5-0b770', 'swiss_bert-0b110', 'llama2-7b', 'legal_xlm_roberta-0b279', 'cerebras-0b111', 'llama-30b', 'cerebras-6b7', 'legal_xlm_roberta-0b561', 'falcon_instruct-40b', 'majority_full_name-1b', 'falcon-40b', 'pythia-12b', 'pythia-1b4', 'flan_t5-3b', 'mt5-13b', 'mt0-13b', 'cerebras-2b7', 'distilbert_squad-0b062', 'bloom-1b7', 'mpt-7b', 'mt5-3b7', 'mpt_instruct-6b7', 'roberta_squad-0b125', 'gptj-6b', 'mt5-1b2', 'incite_instruct-3b', 'mt0-0b300', 'roberta-0b355', 'llama2-70b', 'deberta_squad-0b100', 'flan_t5-11b', 'roberta_squad-0b355', 'mt0-1b2', 'pythia-2b8', 'distilbert-0b066', 'incite-3b', 'falcon-7b', 'llama-65b', 't5-0b220', 'roberta-0b125', 'bloom-1b1', 'flan_t5-0b250', 'flan_t5-0b080', 'mt5-0b580', 'bloomz-3b', 'pythia-0b410', 'xlm_swiss_bert-0b110', 'cerebras-1b3', 'mt0-0b580', 't0-11b', 'gpt_neox-20b', 'bloom-7b1', 'llama2-13b', 'pythia-0b160', 'pythia-6b9', 'mt0-3b7'])
```
