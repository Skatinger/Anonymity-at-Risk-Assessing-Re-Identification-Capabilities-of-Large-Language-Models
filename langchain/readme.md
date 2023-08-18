# Retrieval Augmented Re-Identification

## Installation
1. Install required packages with `pip --install requirements.txt`
## Usage
1. Store the files you would like for retrieval under `data/news-articles-test-set.tsv`
2. Write your OpenAI API key into the shell environment.
3. run `prepare_vector_store.py` to ingest the stored files into a Chroma vector database.
The database will be persisted under `/db`.
4. If you run on court rulings, you can use `paraphrase_hand_picked.py` to create a paraphrased
version alongside the original texts.
5. Run `query.py` (still with OpenAI API key loaded) to run your predictions.

## Costs
Be careful about the number of tokens you use for emebeddings and querying,
as costs can quickly rise. For the embeddings ada-text-002 is used which is cheap,
but for querying GPT-4 and GPT-3.5-16k are used, which are more costly.
GPT-4 has a strict API limit, why after every prediction a pause of 1 minute is introduced.
If these restrictions do not apply to your account you can remove the sleep statement
for much faster processing.