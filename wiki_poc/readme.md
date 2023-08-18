# installation
## Local
1. `python -m venv .env`
2. `source .env/bin/activate`
3. `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
4. `pip install -r requirements.txt`

**Some jobs needs huggingface accelerate and bitsandbytes, check the correct versions for your GPU.**

## Ubelix
When running on a normal machine all scripts can be executed directly. To run on ubelix, slurm jobs have to be defined
as bash scripts. For all scripts different bash scripts are available. Configure to your needs or create new ones.
To create new ones, use the [HPC Documentation](https://hpc-unibe-ch.github.io/) or copy one of `generic-cpu-job.sh`
or `generic-gpu-job.sh`. They serve as templates or to quickly run a short job without overhead.

Before running a job ensure that a python env is configured. Check HPC Documentation on how to do this.
For members of the NRP77 research project, an environment `standard-nlp` is ready to use and set as default in most scripts.

## Scripts for Wiki-POC
Use either the `.py` file locally or on ubelix with your desired options, or run the `.sh` files for default options.
Ensure you load python env and cuda when running without default options.


`download-wiki.py`: downloads the wikipedia dataset required for the downstream tasks, run this first  
`build-paraphrased`: uses the wiki dataset to build collections of paraphrased sentences  
`build-unparaphrased`: the same as build-paraphrased, but keeps the sentences  
`build-masked`: uses the sentences preprocessed in `build-*` and masks entities identified as the person on which 
the wikipedia article is about.

## modules
### models
Run different models with different configurations. Can choose which models, model groups or model sizes.
Add own models by extending the corresponding abstract runners.
- `python -m models.model_runner -h` for explanations and options.

### evaluation
- `python -m evaluation.loader -h`: can load different results and ground truth
- `python -m evaluation.single_prediction_eval <result file-path>`: for quick evals, should be imported and not called directly
- `python -m evaluation.plotting.plotter -h`: builds different plots for given results.

### Good to Know
Texts are split to sentences for paraphrasing only, as this will ensure the paraphrasing only does small changes on the
sentence structure, but not change the full text. Sentences are then joined back together to a text, to improve the
quality of NER, as well as the fill-mask afterwards.

## Scripts and Pipeline for Wiki-Large
1. `prepare-wiki-large`:
    pulls wikipedia dataset and first 700'000 people from wikipedia via sparql query on wikidata. Only keeps wikipedia pages
    which match a person from the 700'000 people found in the sparql query. Saved into `./data`
2. `clean-wiki-large`:
    removes bibliographies and unwanted sections, removes articles shorter than 6'000 characters. Processing might throw some
    "Token indices sequence too long", which is fine. It indicates that the sentence will be truncated. Saved to `./data_reduced`
3. `build-unparaphrased-large`:
    splits raw text of pages into sentences. Saved to `./data_unparaphrased`
4. `create-shards.py`:
    small helper which creates shards of the unparaphrased dataset to allow running several jobs at once when executing
    the paraphrasing. This solves some caching problems and is easier than paralellizing the paraphrasing job.
    Saves shards to `./data_unparaphrased_shard_<index>`
5. `build-paraphrased-large.py <shard-index>`:
    paraphrases sentences to separate dataset column. Takes as argument the number of the shard it should process.
    Final result saved to `./data_paraphrased`
6. `build-masked-large.py`:
    Does named entity recognition on the original and paraphrased texts, and replaces any
    entity matching the person the wiki page is about with a mask token. The string which is replaced by the mask is stored as well, for evaluation. Result is saved to `./data_masked`
